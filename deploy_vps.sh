#!/usr/bin/env bash
#
# Déploiement RwaMarket sur le VPS (Nginx + Docker Compose + SSE).
# Usage :
#   ./deploy_vps.sh
# Variables surchargeables : VPS_HOST, REPO_URL, APP_DIR, DOMAIN, LOCAL_PORT.
#
set -euo pipefail

VPS_HOST="${VPS_HOST:-root@148.230.124.5}"
REPO_URL="${REPO_URL:-https://github.com/keziakdos/rwamarket.git}"
APP_DIR="${APP_DIR:-/opt/rwamarket}"
DOMAIN="${DOMAIN:-rwamarket.rwazap.com}"
LOCAL_PORT="${LOCAL_PORT:-8095}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31mERREUR: %s\033[0m\n' "$*" >&2; exit 1; }

[ -f "${ENV_FILE}" ] || fail "Fichier .env local introuvable : ${ENV_FILE}"
command -v ssh >/dev/null 2>&1 || fail "Le client SSH est requis."
command -v scp >/dev/null 2>&1 || fail "Le client SCP est requis."

log "1/6 - Mise en pause des anciens conteneurs trade/trading/rwatrade"
ssh "${VPS_HOST}" 'bash -s' <<'REMOTE_PAUSE'
set -euo pipefail
paused=0
while read -r container_id container_name; do
  [ -n "${container_id}" ] || continue
  echo "Pause du conteneur ${container_name} (${container_id})"
  docker pause "${container_id}" >/dev/null 2>&1 || true
  paused=$((paused + 1))
done < <(
  docker ps --format '{{.ID}} {{.Names}}' 2>/dev/null \
    | grep -Ei 'trade|trading|rwatrade' \
    | grep -vi 'rwamarket' || true
)
echo "Conteneurs mis en pause : ${paused}"

if [ -d /etc/nginx/sites-enabled ]; then
  while IFS= read -r link; do
    [ -n "${link}" ] || continue
    name="$(basename "${link}")"
    target="$(readlink -f "${link}" 2>/dev/null || true)"
    if printf '%s %s\n' "${name}" "${target}" | grep -qiE 'trade|trading|rwatrade'; then
      echo "Suppression du lien Nginx ${link}"
      rm -f "${link}"
    fi
  done < <(find /etc/nginx/sites-enabled -maxdepth 1 -type l -print 2>/dev/null || true)
fi
REMOTE_PAUSE

log "2/6 - Clone ou mise à jour du dépôt dans ${APP_DIR}"
ssh "${VPS_HOST}" "APP_DIR='${APP_DIR}' REPO_URL='${REPO_URL}' bash -s" <<'REMOTE_GIT'
set -euo pipefail
command -v git >/dev/null 2>&1 || { echo "git absent sur le VPS"; exit 1; }
mkdir -p "$(dirname "${APP_DIR}")"
if [ -d "${APP_DIR}/.git" ]; then
  cd "${APP_DIR}"
  git remote set-url origin "${REPO_URL}"
  git fetch --prune origin
  git checkout main
  git reset --hard origin/main
else
  rm -rf "${APP_DIR}"
  git clone --branch main "${REPO_URL}" "${APP_DIR}"
fi
echo "Commit déployé : $(git -C "${APP_DIR}" rev-parse --short HEAD)"
REMOTE_GIT

log "3/6 - Copie du fichier .env vers le VPS"
ssh "${VPS_HOST}" "mkdir -p '${APP_DIR}'"
scp "${ENV_FILE}" "${VPS_HOST}:${APP_DIR}/.env"
ssh "${VPS_HOST}" "chmod 600 '${APP_DIR}/.env'"

log "4/6 - Build et démarrage Docker Compose (127.0.0.1:${LOCAL_PORT} -> 8000)"
ssh "${VPS_HOST}" "APP_DIR='${APP_DIR}' LOCAL_PORT='${LOCAL_PORT}' bash -s" <<'REMOTE_COMPOSE'
set -euo pipefail
cd "${APP_DIR}"
if docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE="docker-compose"
else
  echo "Docker Compose est introuvable sur le VPS"; exit 1
fi
# Le mapping de port est déjà défini dans docker-compose.yml (127.0.0.1:8095:8000).
if ! grep -q "127.0.0.1:${LOCAL_PORT}:8000" docker-compose.yml 2>/dev/null; then
  echo "Attention : le mapping 127.0.0.1:${LOCAL_PORT}:8000 est absent de docker-compose.yml"
fi
${COMPOSE} up -d --build
${COMPOSE} ps
REMOTE_COMPOSE

log "5/6 - Configuration Nginx (SSE sans buffering)"
ssh "${VPS_HOST}" "cat > '/etc/nginx/sites-available/${DOMAIN}'" <<NGINX
server {
    listen 80;
    listen [::]:80;
    server_name ${DOMAIN};

    client_max_body_size 16m;

    location / {
        proxy_pass http://127.0.0.1:${LOCAL_PORT};
        proxy_http_version 1.1;

        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "";

        # Streaming SSE : aucun buffering ni cache intermédiaire.
        proxy_buffering off;
        proxy_cache off;
        proxy_request_buffering off;
        chunked_transfer_encoding on;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
        gzip off;
    }
}
NGINX

log "6/6 - Activation du site, test et rechargement Nginx"
ssh "${VPS_HOST}" "DOMAIN='${DOMAIN}' bash -s" <<'REMOTE_NGINX'
set -euo pipefail
ln -sfn "/etc/nginx/sites-available/${DOMAIN}" "/etc/nginx/sites-enabled/${DOMAIN}"
nginx -t
systemctl reload nginx
echo "Nginx rechargé pour ${DOMAIN}"
REMOTE_NGINX

log "Déploiement terminé : https://${DOMAIN} (streaming SSE actif)"
