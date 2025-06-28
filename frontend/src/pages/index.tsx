import React, { useState } from 'react';
import { Dashboard } from '../components/Dashboard';
import { CoinAnalysis } from '../components/CoinAnalysis';
import { MLProgress } from '../components/MLProgress';
import { StrategyConfigurator } from '../components/StrategyConfigurator';
import { StrategyConfig } from '../components/StrategyConfig';
import { TradingSignals } from '../components/TradingSignals';
import { TrainingStatus } from '../components/TrainingStatus';
import { StrategyOptimizer } from '../components/StrategyOptimizer';
import { ModernTradingDashboard } from '../components/ModernTradingDashboard';
import { TradingControl } from '../components/TradingControl';
import { StatusTopBar } from '../components/StatusTopBar';

export default function Home() {
  const [activeTab, setActiveTab] = useState(0);

  const tabs = [
    { id: 0, label: '🚀 Trading Dashboard', component: <ModernTradingDashboard onNavigate={setActiveTab} /> },
    { id: 1, label: '🎮 Trading Control', component: <TradingControl /> },
    { id: 2, label: '📊 ML Progress', component: <MLProgress /> },
    { id: 3, label: '🔧 Training Status', component: <TrainingStatus /> },
    { id: 4, label: '📡 Signals', component: <TradingSignals /> },
    { id: 5, label: '⚙️ Strategy Config', component: <StrategyConfig /> },
    { id: 6, label: '🤖 AI Optimizer', component: <StrategyOptimizer /> },
    { id: 7, label: '📈 Coin Analysis', component: <CoinAnalysis /> }
  ];

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Status Top Bar - Always Visible */}
      <StatusTopBar />

      {/* Navigation Menu - Full Width */}
      <div className="bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900 border-b border-gray-700/50">
        <div className="flex items-center px-6 py-2 overflow-x-auto">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-6 py-3 mx-2 rounded-lg font-semibold transition-all duration-300 whitespace-nowrap ${
                activeTab === tab.id
                  ? 'bg-gradient-to-r from-cyan-500/30 to-purple-500/30 border border-cyan-500/50 text-cyan-400 shadow-[0_0_20px_rgba(0,255,255,0.3)]'
                  : 'bg-black/30 border border-gray-700/50 text-gray-400 hover:bg-gray-700/30 hover:text-gray-300 hover:border-gray-600/50'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content Area */}
      <div className="relative">
        {tabs[activeTab]?.component}
      </div>
    </div>
  );
}