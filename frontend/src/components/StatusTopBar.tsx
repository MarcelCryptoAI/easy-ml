import React from 'react';
import { 
  Storage, 
  SmartToy, 
  TrendingUp, 
  AccountBalance,
  Refresh,
  Computer,
  Api,
  Memory
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { tradingApi } from '../utils/api';

interface ConnectionStatus {
  frontend_connected: boolean;
  backend_connected: boolean;
  worker_connected: boolean;
  database_connected: boolean;
  openai_connected: boolean;
  bybit_connected: boolean;
  uta_balance: string;
}

export const StatusTopBar: React.FC = () => {
  const { data: statusData, refetch } = useQuery({
    queryKey: ['system-status'],
    queryFn: async () => {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout
      
      try {
        // Try new status endpoint first
        const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/status`, {
          signal: controller.signal
        });
        clearTimeout(timeoutId);
        
        if (response.ok) {
          const data = await response.json();
          return data as ConnectionStatus;
        }
      } catch (error) {
        clearTimeout(timeoutId);
        console.warn('Status endpoint failed, falling back to health endpoint');
      }
      
      try {
        // Fallback to health endpoint
        const healthResponse = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/health`, {
          signal: controller.signal
        });
        clearTimeout(timeoutId);
        const healthData = await healthResponse.json();
        
        // Transform health data to status format
        return {
          frontend_connected: true,
          backend_connected: true,
          worker_connected: true, // Assume true if backend is running
          database_connected: healthData.database_coins > 0,
          openai_connected: true, // Assume true if backend is running
          bybit_connected: healthData.bybit_connected,
          uta_balance: healthData.account_balance?.toString() || "0.00"
        } as ConnectionStatus;
      } catch (error) {
        clearTimeout(timeoutId);
        // Return default disconnected state
        return {
          frontend_connected: true,
          backend_connected: false,
          worker_connected: false,
          database_connected: false,
          openai_connected: false,
          bybit_connected: false,
          uta_balance: "0.00"
        } as ConnectionStatus;
      }
    },
    refetchInterval: 15000, // Update every 15 seconds (less frequent)
    retry: 2
  });

  const getStatusIndicator = (connected: boolean, label: string, icon: React.ReactNode) => (
    <div className="flex items-center gap-2 px-3 py-2 bg-black/30 backdrop-blur-sm border border-gray-700/50 rounded-lg">
      <div className="flex items-center gap-1">
        {icon}
        <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`}></div>
      </div>
      <span className={`text-sm font-medium ${connected ? 'text-green-400' : 'text-red-400'}`}>
        {label}
      </span>
    </div>
  );

  return (
    <div className="bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900 border-b border-gray-700/50 backdrop-blur-xl">
      <div className="flex items-center justify-between px-6 py-3">
        {/* Title */}
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-400">
            🤖 Crypto Trading ML Platform
          </h1>
          <div className="px-3 py-1 bg-gradient-to-r from-green-500/20 to-emerald-500/20 border border-green-500/50 rounded-lg">
            <span className="text-green-400 text-sm font-semibold">LIVE AUTONOMOUS MODE</span>
          </div>
        </div>

        {/* Connection Status & Balance */}
        <div className="flex items-center gap-3">
          {/* Status Indicators */}
          {getStatusIndicator(statusData?.frontend_connected || true, 'Frontend', <Computer className="w-4 h-4 text-cyan-400" />)}
          {getStatusIndicator(statusData?.backend_connected || false, 'Backend', <Api className="w-4 h-4 text-purple-400" />)}
          {getStatusIndicator(statusData?.worker_connected || false, 'Worker', <Memory className="w-4 h-4 text-blue-400" />)}
          {getStatusIndicator(statusData?.database_connected || false, 'Database', <Storage className="w-4 h-4 text-indigo-400" />)}
          {getStatusIndicator(statusData?.bybit_connected || false, 'ByBit API', <TrendingUp className="w-4 h-4 text-yellow-400" />)}
          {getStatusIndicator(statusData?.openai_connected || false, 'OpenAI API', <SmartToy className="w-4 h-4 text-pink-400" />)}

          {/* Account Balance */}
          <div className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-green-500/20 to-emerald-500/20 border border-green-500/50 rounded-lg">
            <AccountBalance className="w-5 h-5 text-green-400" />
            <span className="text-green-400 font-bold text-lg">
              ${parseFloat(statusData?.uta_balance || '0').toFixed(2)}
            </span>
          </div>

          {/* Refresh Button */}
          <button
            onClick={() => refetch()}
            className="p-2 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 border border-cyan-500/50 rounded-lg text-cyan-400 hover:bg-cyan-500/30 transition-all duration-300 hover:scale-105"
          >
            <Refresh className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
};