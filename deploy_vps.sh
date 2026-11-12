#!/usr/bin/env bash
set -e

VPS_USER="trading"
VPS_HOST="148.230.124.5"
REMOTE_DIR="/opt/rwamarket"

echo "1. Envoi du fichier .env local vers le VPS..."
scp .env ${VPS_USER}@${VPS_HOST}:/tmp/rwamarket.env

echo "2. Mise à jour du code et redémarrage Docker sur le VPS..."
ssh -t ${VPS_USER}@${VPS_HOST} "
  sudo mkdir -p ${REMOTE_DIR}/data &&
  sudo chown -R ${VPS_USER}:${VPS_USER} ${REMOTE_DIR} &&
  cd ${REMOTE_DIR} &&
  git pull origin main &&
  mv /tmp/rwamarket.env ${REMOTE_DIR}/.env &&
  sudo docker compose up -d --build &&
  sudo docker compose ps &&
  sudo nginx -t &&
  sudo systemctl reload nginx
"

echo "Déploiement terminé sur https://rwamarket.rwazap.com"
