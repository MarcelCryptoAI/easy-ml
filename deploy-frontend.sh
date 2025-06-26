#!/bin/bash

# Navigeer naar frontend directory
cd frontend

# Installeer dependencies
npm install

# Build de applicatie
npm run build

# Start de applicatie (voor lokale test)
echo "Frontend built successfully!"
echo "You can now deploy this to Railway manually via the web interface."
echo ""
echo "Steps to deploy:"
echo "1. Go to https://railway.app/dashboard"
echo "2. Open your 'easy-ml' project"
echo "3. Click 'New Service' > 'GitHub Repo'"
echo "4. Upload the frontend folder to GitHub first, then connect it"
echo ""
echo "Backend URL: https://easy-ml-production.up.railway.app"
echo ""
echo "Or use Railway template deployment:"
echo "1. Zip the frontend folder"
echo "2. Go to railway.app/new"
echo "3. Deploy from template"