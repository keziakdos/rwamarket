#!/usr/bin/env bash
#
# Déploiement RwaMarket sur le VPS (Docker Compose + Nginx + SSE).
# Usage : ./deploy_vps.sh
#
set -euo pipefail

VPS_USER="${VPS_USER:-trading}"
VPS_HOST="${VPS_HOST:-148.230.124.5}"
REMOTE_DIR="${REMOTE_DIR:-/opt/rwamarket}"
REPO_URL="${REPO_URL:-https://github.com/keziakdos/rwamarket.git}"
DOMAIN="${DOMAIN:-rwamarket.rwazap.com}"
LOCAL_PORT="${LOCAL_PORT:-8096}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31mERREUR: %s\033[0m\n' "$*" >&2; exit 1; }

[ -f "${ENV_FILE}" ] || fail "Fichier .env local introuvable : ${ENV_FILE}"
# Garde-fou : un .env corrompu (ligne de heredoc collée) casse Docker Compose.
if head -n 1 "${ENV_FILE}" | grep -q 'cat <<'; then
  fail "Le .env local commence par une ligne de heredoc. Corrigez-le avant de déployer."
fi
if grep -qx 'EOF' "${ENV_FILE}"; then
  fail "Le .env local contient une ligne EOF parasite. Corrigez-le avant de déployer."
fi

command -v ssh >/dev/null 2>&1 || fail "Le client SSH est requis."
command -v scp >/dev/null 2>&1 || fail "Le client SCP est requis."

log "1/5 - Envoi du .env vers ${VPS_USER}@${VPS_HOST}"
scp "${ENV_FILE}" "${VPS_USER}@${VPS_HOST}:/tmp/rwamarket.env"

log "2/5 - Mise à jour du code, build et démarrage Docker (port ${LOCAL_PORT})"
ssh -t "${VPS_USER}@${VPS_HOST}" \
  "REMOTE_DIR='${REMOTE_DIR}' REPO_URL='${REPO_URL}' LOCAL_PORT='${LOCAL_PORT}' bash -s" <<'REMOTE'
set -euo pipefail
sudo mkdir -p "${REMOTE_DIR}/data"
sudo chown -R "$(id -u):$(id -g)" "${REMOTE_DIR}"

if [ -d "${REMOTE_DIR}/.git" ]; then
  git -C "${REMOTE_DIR}" remote set-url origin "${REPO_URL}"
  git -C "${REMOTE_DIR}" fetch --prune origin
  git -C "${REMOTE_DIR}" checkout main
  git -C "${REMOTE_DIR}" reset --hard origin/main
else
  rm -rf "${REMOTE_DIR}"
  git clone --branch main "${REPO_URL}" "${REMOTE_DIR}"
fi

mv /tmp/rwamarket.env "${REMOTE_DIR}/.env"
chmod 600 "${REMOTE_DIR}/.env"

if docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE="docker-compose"
else
  echo "Docker Compose introuvable sur le VPS"; exit 1
fi

cd "${REMOTE_DIR}"
# Le port est défini dans docker-compose.yml ; on le vérifie pour éviter un conflit.
grep -q "127.0.0.1:${LOCAL_PORT}:8000" docker-compose.yml \
  || { echo "Mapping 127.0.0.1:${LOCAL_PORT}:8000 absent de docker-compose.yml"; exit 1; }

sudo ${COMPOSE} up -d --build
sudo ${COMPOSE} ps
curl -sS -m 10 -o /dev/null -w "Local ${LOCAL_PORT} -> HTTP %{http_code}\n" "http://127.0.0.1:${LOCAL_PORT}/login"
REMOTE

log "3/5 - Configuration Nginx (proxy ${LOCAL_PORT} + SSE sans buffering)"
ssh -t "${VPS_USER}@${VPS_HOST}" \
  "DOMAIN='${DOMAIN}' LOCAL_PORT='${LOCAL_PORT}' bash -s" <<'REMOTE_NGINX'
set -euo pipefail
sudo tee "/etc/nginx/sites-available/${DOMAIN}" >/dev/null <<NGINX
server {
    listen 443 ssl;
    listen [::]:443 ssl;
    server_name ${DOMAIN};

    ssl_certificate /etc/letsencrypt/live/${DOMAIN}/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/${DOMAIN}/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;

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
        proxy_buffering off;
        proxy_cache off;
        proxy_request_buffering off;
        chunked_transfer_encoding on;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
        gzip off;
    }
}

server {
    listen 80;
    listen [::]:80;
    server_name ${DOMAIN};
    if (\$host = ${DOMAIN}) {
        return 301 https://\$host\$request_uri;
    }
    return 404;
}
NGINX

sudo ln -sfn "/etc/nginx/sites-available/${DOMAIN}" "/etc/nginx/sites-enabled/${DOMAIN}"
sudo nginx -t
sudo systemctl reload nginx
echo "Nginx rechargé pour ${DOMAIN}"
REMOTE_NGINX

log "4/5 - Vérification publique"
curl -sS -m 20 -o /dev/null -w "https://${DOMAIN}/login -> HTTP %{http_code}\n" \
  "https://${DOMAIN}/login"

log "5/5 - Déploiement terminé : https://${DOMAIN}"
