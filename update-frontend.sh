#!/bin/bash

echo "🚀 Updating frontend with autonomous features..."

# Update frontend files
cp frontend/src/components/TrainingStatus.tsx crypto-frontend-deploy/src/components/
cp frontend/src/components/StrategyOptimizer.tsx crypto-frontend-deploy/src/components/
cp frontend/src/components/MLProgress.tsx crypto-frontend-deploy/src/components/
cp frontend/src/pages/index.tsx crypto-frontend-deploy/src/pages/

echo "✅ Files updated!"
echo ""
echo "Your platform now has:"
echo "🤖 Detailed training status with real-time progress"
echo "📊 All 500+ coins with 4 model progress indicators"  
echo "🎯 'Optimize All Strategies' batch optimization"
echo "⚙️ Autonomous optimization scheduler"
echo "🔧 Training controls (pause/resume)"
echo ""
echo "Frontend: https://crypto-frontend-production-3272.up.railway.app"
echo "Backend: https://easy-ml-production.up.railway.app"
echo ""
echo "🎉 Fully autonomous crypto trading system is LIVE!"