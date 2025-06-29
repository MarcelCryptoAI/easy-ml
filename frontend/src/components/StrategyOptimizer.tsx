import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { tradingApi } from '../utils/api';
import toast from 'react-hot-toast';

interface OptimizationJob {
  coin_symbol: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  progress: number;
  current_params: {
    take_profit_percentage: number;
    stop_loss_percentage: number;
    leverage: number;
  };
  optimized_params?: {
    take_profit_percentage: number;
    stop_loss_percentage: number;
    leverage: number;
  };
  backtest_results?: {
    total_return: number;
    win_rate: number;
    max_drawdown: number;
    sharpe_ratio: number;
    total_trades: number;
    profit_factor: number;
  };
  improvement_percentage?: number;
  started_at?: string;
  completed_at?: string;
  queue_position: number;
}

interface OptimizationSession {
  is_running: boolean;
  total_coins: number;
  completed_coins: number;
  current_coin: string;
  session_start_time: string;
  estimated_completion_time: string;
  auto_apply_optimizations: boolean;
}

export const StrategyOptimizer: React.FC = () => {
  const [showSettingsDialog, setShowSettingsDialog] = useState(false);
  const [autoOptimizeSettings, setAutoOptimizeSettings] = useState({
    enabled: false,
    interval_hours: 24,
    min_improvement_threshold: 5.0,
    auto_apply: true
  });

  const queryClient = useQueryClient();

  const { data: coins = [] } = useQuery({
    queryKey: ['coins'],
    queryFn: tradingApi.getCoins
  });

  const { data: optimizationSession, refetch: refetchSession } = useQuery({
    queryKey: ['optimization-session'],
    queryFn: async () => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/optimize/status`);
      if (!response.ok) throw new Error('Failed to fetch optimization session');
      return await response.json() as OptimizationSession;
    },
    refetchInterval: 5000
  });

  const { data: optimizationQueue = [] } = useQuery({
    queryKey: ['optimization-queue'],
    queryFn: async () => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/optimize/queue`);
      if (!response.ok) throw new Error('Failed to fetch optimization queue');
      return await response.json() as OptimizationJob[];
    },
    refetchInterval: 3000
  });

  const startOptimizeAllMutation = useMutation({
    mutationFn: async () => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/optimize/batch-all`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          auto_apply: autoOptimizeSettings.auto_apply,
          min_improvement_threshold: autoOptimizeSettings.min_improvement_threshold
        })
      });
      return response.json();
    },
    onSuccess: () => {
      toast.success('Started optimizing all strategies!');
      refetchSession();
    },
    onError: () => {
      toast.error('Failed to start optimization');
    }
  });

  const stopOptimizationMutation = useMutation({
    mutationFn: async () => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/optimize/stop`, {
        method: 'POST'
      });
      return response.json();
    },
    onSuccess: () => {
      toast.success('Optimization stopped');
      refetchSession();
    }
  });

  const enableAutoOptimizeMutation = useMutation({
    mutationFn: async (settings: typeof autoOptimizeSettings) => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/optimize/auto-schedule`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
      });
      return response.json();
    },
    onSuccess: () => {
      toast.success('Auto-optimization configured!');
      setShowSettingsDialog(false);
    }
  });

  const createDefaultStrategiesMutation = useMutation({
    mutationFn: async () => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/strategies/create-defaults`, {
        method: 'POST'
      });
      return response.json();
    },
    onSuccess: () => {
      toast.success('Default strategies created for all coins!');
    }
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'warning';
      case 'completed': return 'success';
      case 'pending': return 'default';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  const formatTime = (isoString: string) => {
    if (!isoString) return '-';
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = date.getTime() - now.getTime();
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    
    if (diffHours > 24) {
      return `${Math.floor(diffHours / 24)} days`;
    } else {
      return `${diffHours} hours`;
    }
  };

  return (
    <div className="min-h-screen bg-black text-white relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-gradient-to-br from-orange-900/20 via-black to-yellow-900/20" />
        <div className="absolute top-0 left-0 w-96 h-96 bg-orange-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse" />
        <div className="absolute bottom-0 right-0 w-96 h-96 bg-yellow-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse" />
      </div>

      <div className="relative z-10 p-8">
        {/* Header */}
        <h1 className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-orange-400 via-yellow-500 to-amber-500 mb-8">
          🎯 AI Strategy Optimizer
        </h1>

        {/* Main Control Panel */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Bulk Strategy Optimization */}
          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-orange-500/20 to-yellow-500/20 rounded-2xl blur-xl" />
            <div className="relative bg-black/50 backdrop-blur-xl border border-orange-500/30 rounded-2xl p-6">
              <h3 className="text-2xl font-bold text-orange-400 mb-4">Bulk Strategy Optimization</h3>
              
              {optimizationSession?.is_running ? (
                <div className="space-y-4">
                  <p className="text-gray-400">
                    Optimizing: {optimizationSession.current_coin}
                  </p>
                  <div className="relative">
                    <div className="w-full bg-gray-700 rounded-full h-4">
                      <div 
                        className="bg-gradient-to-r from-orange-400 to-yellow-500 h-4 rounded-full transition-all duration-500"
                        style={{ width: `${(optimizationSession.completed_coins / optimizationSession.total_coins) * 100}%` }}
                      ></div>
                    </div>
                    <span className="absolute inset-0 flex items-center justify-center text-xs font-semibold text-white">
                      {((optimizationSession.completed_coins / optimizationSession.total_coins) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <p className="text-sm text-gray-400">
                    {optimizationSession.completed_coins} / {optimizationSession.total_coins} coins completed
                  </p>
                  <button
                    onClick={() => stopOptimizationMutation.mutate()}
                    disabled={stopOptimizationMutation.isPending}
                    className="w-full px-6 py-3 bg-gradient-to-r from-red-500/20 to-pink-500/20 backdrop-blur-xl border border-red-500/50 rounded-xl text-red-400 hover:bg-red-500/30 transition-all duration-300 disabled:opacity-50"
                  >
                    <div className="flex items-center justify-center gap-2">
                      <span>⏹️</span>
                      <span>Stop Optimization</span>
                    </div>
                  </button>
                </div>
              ) : (
                <div className="space-y-4">
                  <p className="text-gray-400">
                    Optimize strategies for all {coins.length} coins using AI and backtesting
                  </p>
                  <div className="flex flex-wrap gap-3">
                    <button
                      onClick={() => startOptimizeAllMutation.mutate()}
                      disabled={startOptimizeAllMutation.isPending}
                      className="px-6 py-3 bg-gradient-to-r from-orange-500/20 to-yellow-500/20 backdrop-blur-xl border border-orange-500/50 rounded-xl text-orange-400 hover:bg-orange-500/30 transition-all duration-300 hover:scale-105 disabled:opacity-50"
                    >
                      <div className="flex items-center gap-2">
                        {startOptimizeAllMutation.isPending ? (
                          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-orange-400"></div>
                        ) : (
                          <span>✨</span>
                        )}
                        <span>Optimize All Strategies</span>
                      </div>
                    </button>
                    <button
                      onClick={() => setShowSettingsDialog(true)}
                      className="px-4 py-3 bg-gradient-to-r from-purple-500/20 to-pink-500/20 backdrop-blur-xl border border-purple-500/50 rounded-xl text-purple-400 hover:bg-purple-500/30 transition-all duration-300"
                    >
                      <div className="flex items-center gap-2">
                        <span>⚙️</span>
                        <span>Settings</span>
                      </div>
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Autonomous Optimization */}
          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-cyan-500/20 rounded-2xl blur-xl" />
            <div className="relative bg-black/50 backdrop-blur-xl border border-blue-500/30 rounded-2xl p-6">
              <h3 className="text-2xl font-bold text-blue-400 mb-4">Autonomous Optimization</h3>
              <p className="text-gray-400 mb-6">
                Automatically optimize strategies every {autoOptimizeSettings.interval_hours} hours
              </p>
              
              <div className="flex items-center justify-between mb-6">
                <span className="text-gray-300">Enable Auto-Optimization</span>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={autoOptimizeSettings.enabled}
                    onChange={(e) => setAutoOptimizeSettings({
                      ...autoOptimizeSettings,
                      enabled: e.target.checked
                    })}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                </label>
              </div>
              
              <button
                onClick={() => setShowSettingsDialog(true)}
                className="w-full px-4 py-3 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 backdrop-blur-xl border border-cyan-500/50 rounded-xl text-cyan-400 hover:bg-cyan-500/30 transition-all duration-300"
              >
                <div className="flex items-center justify-center gap-2">
                  <span>📅</span>
                  <span>Configure Schedule</span>
                </div>
              </button>
            </div>
          </div>
        </div>

        {/* Default Strategies Setup */}
        <div className="relative mb-8">
          <div className="absolute inset-0 bg-gradient-to-r from-green-500/20 to-emerald-500/20 rounded-2xl blur-xl" />
          <div className="relative bg-black/50 backdrop-blur-xl border border-green-500/30 rounded-2xl p-6">
            <h3 className="text-2xl font-bold text-green-400 mb-4">Default Strategy Setup</h3>
            <p className="text-gray-400 mb-6">
              Create default trading strategies for all coins that don't have one yet
            </p>
            <button
              onClick={() => createDefaultStrategiesMutation.mutate()}
              disabled={createDefaultStrategiesMutation.isPending}
              className="px-6 py-3 bg-gradient-to-r from-green-500/20 to-emerald-500/20 backdrop-blur-xl border border-green-500/50 rounded-xl text-green-400 hover:bg-green-500/30 transition-all duration-300 disabled:opacity-50"
            >
              <div className="flex items-center gap-2">
                {createDefaultStrategiesMutation.isPending ? (
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-green-400"></div>
                ) : (
                  <span>▶️</span>
                )}
                <span>Create Default Strategies</span>
              </div>
            </button>
          </div>
        </div>

        {/* Optimization Queue */}
        {optimizationQueue.length > 0 && (
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-gray-500/10 to-gray-500/10 rounded-2xl blur-xl" />
            <div className="relative bg-black/50 backdrop-blur-xl border border-gray-500/30 rounded-2xl overflow-hidden">
              <div className="p-6 border-b border-gray-700">
                <h3 className="text-2xl font-bold text-gray-100">
                  Optimization Queue ({optimizationQueue.length} coins)
                </h3>
              </div>
              
              <div className="p-6">
                <div className="overflow-x-auto">
                  <table className="w-full text-white">
                    <thead>
                      <tr className="border-b border-gray-700">
                        <th className="text-left p-4 text-cyan-400 font-bold">Position</th>
                        <th className="text-left p-4 text-cyan-400 font-bold">Coin</th>
                        <th className="text-left p-4 text-cyan-400 font-bold">Status</th>
                        <th className="text-left p-4 text-cyan-400 font-bold">Progress</th>
                        <th className="text-left p-4 text-cyan-400 font-bold">Expected Improvement</th>
                        <th className="text-left p-4 text-cyan-400 font-bold">Started</th>
                      </tr>
                    </thead>
                    <tbody>
                      {optimizationQueue.map((job, index) => (
                        <tr key={job.coin_symbol} className="border-b border-gray-800 hover:bg-gradient-to-r hover:from-orange-500/10 hover:to-transparent transition-all duration-300">
                          <td className="p-4">
                            <span className="font-bold text-white">#{job.queue_position}</span>
                          </td>
                          <td className="p-4">
                            <span className="font-bold text-white">{job.coin_symbol}</span>
                          </td>
                          <td className="p-4">
                            <div className={`inline-flex items-center px-3 py-1 rounded-lg ${
                              job.status === 'completed' ? 'bg-gradient-to-r from-green-500/20 to-emerald-500/20 border border-green-500/50' :
                              job.status === 'running' ? 'bg-gradient-to-r from-yellow-500/20 to-orange-500/20 border border-yellow-500/50' :
                              job.status === 'error' ? 'bg-gradient-to-r from-red-500/20 to-pink-500/20 border border-red-500/50' :
                              'bg-gradient-to-r from-gray-500/20 to-gray-500/20 border border-gray-500/50'
                            }`}>
                              <span className={`font-semibold text-sm ${
                                job.status === 'completed' ? 'text-green-400' :
                                job.status === 'running' ? 'text-yellow-400' :
                                job.status === 'error' ? 'text-red-400' :
                                'text-gray-400'
                              }`}>
                                {job.status}
                              </span>
                            </div>
                          </td>
                          <td className="p-4">
                            <div className="w-24">
                              <div className="w-full bg-gray-700 rounded-full h-2">
                                <div 
                                  className="bg-gradient-to-r from-orange-400 to-yellow-500 h-2 rounded-full transition-all duration-500"
                                  style={{ width: `${job.progress}%` }}
                                ></div>
                              </div>
                              <span className="text-xs text-gray-400 mt-1">{job.progress}%</span>
                            </div>
                          </td>
                          <td className="p-4">
                            {job.improvement_percentage && (
                              <div className={`inline-flex items-center gap-1 px-3 py-1 rounded-lg ${
                                job.improvement_percentage > 0 
                                  ? 'bg-gradient-to-r from-green-500/20 to-emerald-500/20 border border-green-500/50' 
                                  : 'bg-gradient-to-r from-red-500/20 to-pink-500/20 border border-red-500/50'
                              }`}>
                                <span className="text-sm">{job.improvement_percentage > 0 ? '📈' : '📉'}</span>
                                <span className={`font-semibold text-sm ${
                                  job.improvement_percentage > 0 ? 'text-green-400' : 'text-red-400'
                                }`}>
                                  {job.improvement_percentage > 0 ? '+' : ''}{job.improvement_percentage.toFixed(1)}%
                                </span>
                              </div>
                            )}
                          </td>
                          <td className="p-4 text-gray-400 text-sm">
                            {job.started_at ? new Date(job.started_at).toLocaleTimeString() : '-'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Settings Modal */}
        {showSettingsDialog && (
          <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="relative max-w-md w-full">
              <div className="absolute inset-0 bg-gradient-to-r from-orange-500/20 to-yellow-500/20 rounded-2xl blur-xl" />
              <div className="relative bg-black/90 backdrop-blur-xl border border-orange-500/30 rounded-2xl p-6">
                <h3 className="text-2xl font-bold text-orange-400 mb-6">Auto-Optimization Settings</h3>
                
                <div className="space-y-6">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Enable Autonomous Optimization</span>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        checked={autoOptimizeSettings.enabled}
                        onChange={(e) => setAutoOptimizeSettings({
                          ...autoOptimizeSettings,
                          enabled: e.target.checked
                        })}
                        className="sr-only peer"
                      />
                      <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-orange-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-orange-600"></div>
                    </label>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm mb-2">Optimization Interval (hours)</label>
                    <input
                      type="number"
                      value={autoOptimizeSettings.interval_hours}
                      onChange={(e) => setAutoOptimizeSettings({
                        ...autoOptimizeSettings,
                        interval_hours: parseInt(e.target.value)
                      })}
                      className="w-full px-3 py-2 bg-black/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-orange-500"
                    />
                    <p className="text-xs text-gray-400 mt-1">How often to automatically optimize all strategies</p>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm mb-2">Minimum Improvement Threshold (%)</label>
                    <input
                      type="number"
                      step="0.1"
                      value={autoOptimizeSettings.min_improvement_threshold}
                      onChange={(e) => setAutoOptimizeSettings({
                        ...autoOptimizeSettings,
                        min_improvement_threshold: parseFloat(e.target.value)
                      })}
                      className="w-full px-3 py-2 bg-black/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-orange-500"
                    />
                    <p className="text-xs text-gray-400 mt-1">Only apply optimizations with at least this much improvement</p>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Automatically Apply Optimizations</span>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        checked={autoOptimizeSettings.auto_apply}
                        onChange={(e) => setAutoOptimizeSettings({
                          ...autoOptimizeSettings,
                          auto_apply: e.target.checked
                        })}
                        className="sr-only peer"
                      />
                      <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-orange-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-orange-600"></div>
                    </label>
                  </div>
                </div>

                <div className="flex gap-3 mt-8">
                  <button
                    onClick={() => setShowSettingsDialog(false)}
                    className="flex-1 px-4 py-2 bg-gradient-to-r from-gray-500/20 to-gray-500/20 backdrop-blur-xl border border-gray-500/50 rounded-xl text-gray-400 hover:bg-gray-500/30 transition-all duration-300"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={() => enableAutoOptimizeMutation.mutate(autoOptimizeSettings)}
                    disabled={enableAutoOptimizeMutation.isPending}
                    className="flex-1 px-4 py-2 bg-gradient-to-r from-orange-500/20 to-yellow-500/20 backdrop-blur-xl border border-orange-500/50 rounded-xl text-orange-400 hover:bg-orange-500/30 transition-all duration-300 disabled:opacity-50"
                  >
                    Save Settings
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};