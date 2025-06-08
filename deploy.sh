#!/bin/bash

# AI Mechanic Deployment Script
# Run this on your Ubuntu/Debian server

set -e

echo "🚀 Deploying AI Mechanic to server..."

# Update system
sudo apt update
sudo apt install -y python3 python3-pip nginx nodejs npm

# Create application directory
sudo mkdir -p /opt/ai-mechanic
sudo chown $USER:$USER /opt/ai-mechanic
cd /opt/ai-mechanic

# Copy your files here (you'll need to upload them first)
# scp -r "trackit AI"/* user@your-server:/opt/ai-mechanic/

# Install Python dependencies
pip3 install -r requirements.txt

# Install PM2 for process management
sudo npm install -g pm2

# Create systemd service for the API
sudo tee /etc/systemd/system/ai-mechanic-api.service > /dev/null <<EOF
[Unit]
Description=AI Mechanic API
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/ai-mechanic
Environment=PATH=/usr/bin:/usr/local/bin
Environment=PYTHONPATH=/opt/ai-mechanic
ExecStart=/usr/bin/python3 /opt/ai-mechanic/api_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Configure Nginx
sudo tee /etc/nginx/sites-available/ai-mechanic > /dev/null <<EOF
server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain

    # Serve React build
    location / {
        root /opt/ai-mechanic/build;
        try_files \$uri \$uri/ /index.html;
    }

    # Proxy API requests
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # API docs
    location /docs {
        proxy_pass http://localhost:8000/docs;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOF

# Enable the site
sudo ln -sf /etc/nginx/sites-available/ai-mechanic /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test nginx config
sudo nginx -t

# Start services
sudo systemctl daemon-reload
sudo systemctl enable ai-mechanic-api
sudo systemctl start ai-mechanic-api
sudo systemctl enable nginx
sudo systemctl restart nginx

echo "✅ Deployment complete!"
echo "🌐 Your AI Mechanic should be available at http://your-domain.com"
echo "📍 API: http://your-domain.com/api/"
echo "📍 Docs: http://your-domain.com/docs"