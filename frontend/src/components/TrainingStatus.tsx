import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { tradingApi } from '../utils/api';
import toast from 'react-hot-toast';

interface TrainingQueueItem {
  coin_symbol: string;
  model_type: string;
  status: 'pending' | 'training' | 'completed' | 'error';
  progress: number;
  estimated_time_remaining: number;
  started_at?: string;
  completed_at?: string;
  queue_position: number;
}

interface TrainingSession {
  current_coin: string;
  current_model: string;
  progress: number;
  overall_progress: number;
  eta_seconds: number;
  total_queue_items: number;
  completed_items: number;
  remaining_models: number;
  session_start_time: string;
  estimated_completion_time: string;
}

export const TrainingStatus: React.FC = () => {
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState({
    autoRetrain: true,
    trainingInterval: 3600, // seconds
    maxModelsPerCoin: 10,
    enableNotifications: true,
    batchSize: 5
  });
  const queryClient = useQueryClient();

  // Use the optimized training-info endpoint for consistency
  const { data: trainingInfo, isLoading: trainingLoading, refetch: refetchTraining } = useQuery({
    queryKey: ['training-info'],
    queryFn: () => tradingApi.getTrainingInfo(),
    refetchInterval: autoRefresh ? 5000 : false,
    staleTime: 2000 // Consider data fresh for 2 seconds
  });

  // Get detailed statistics for better insights
  const { data: trainingStats, isLoading: statsLoading, refetch: refetchStats } = useQuery({
    queryKey: ['training-statistics'],
    queryFn: async () => {
      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/training/statistics`);
        if (!response.ok) throw new Error('Failed to fetch statistics');
        return response.json();
      } catch (error) {
        console.error('Error fetching training statistics:', error);
        return { model_statistics: [], top_coins: [], recent_activity: [] };
      }
    },
    refetchInterval: autoRefresh ? 10000 : false,
    staleTime: 5000 // Statistics can be stale for longer
  });

  // Transform training info to match expected interface for backward compatibility
  const trainingSession = trainingInfo ? {
    current_coin: trainingInfo.current_coin,
    current_model: trainingInfo.current_model,
    progress: trainingInfo.current_model_progress || 0,
    status: trainingInfo.status,
    total_coins: trainingInfo.total_coins,
    completed_predictions: trainingInfo.completed_predictions,
    overall_progress: trainingInfo.overall_percentage,
    completed_items: trainingInfo.completed_predictions || 0,
    total_queue_items: trainingInfo.total_coins || 0,
    remaining_models: (trainingInfo.total_coins || 0) - (trainingInfo.completed_predictions || 0),
    eta_seconds: trainingInfo.eta_seconds || 0,
    session_start_time: trainingInfo.session_start_time || new Date().toISOString(),
    estimated_completion_time: trainingInfo.estimated_completion_time || new Date().toISOString()
  } : null;

  const trainingQueue = trainingStats?.recent_activity?.map((activity: any, index: number) => ({
    coin_symbol: activity.coin_symbol,
    model_type: activity.model_type,
    status: 'completed' as const,
    progress: 100,
    estimated_time_remaining: 0,
    completed_at: activity.created_at,
    queue_position: index + 1
  })) || [];

  const sessionLoading = trainingLoading;
  const queueLoading = statsLoading;

  const pauseTrainingMutation = useMutation({
    mutationFn: async () => {
      // API call to pause training
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/training/pause`, {
        method: 'POST'
      });
      if (!response.ok) throw new Error('Failed to pause training');
      return response.json();
    },
    onSuccess: () => {
      toast.success('Training paused');
      refetchTraining();
      refetchStats();
    },
    onError: () => {
      toast.error('Failed to pause training');
    }
  });

  const resumeTrainingMutation = useMutation({
    mutationFn: async () => {
      // API call to resume training
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/training/resume`, {
        method: 'POST'
      });
      if (!response.ok) throw new Error('Failed to resume training');
      return response.json();
    },
    onSuccess: () => {
      toast.success('Training resumed');
      refetchTraining();
      refetchStats();
    },
    onError: () => {
      toast.error('Failed to resume training');
    }
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'training': return 'warning';
      case 'completed': return 'success';
      case 'pending': return 'default';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  };

  const formatEstimatedCompletion = (isoString: string) => {
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = date.getTime() - now.getTime();
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffHours / 24);
    
    if (diffDays > 0) {
      return `${diffDays} days, ${diffHours % 24} hours`;
    } else {
      return `${diffHours} hours`;
    }
  };

  return (
    <div className="min-h-screen bg-black text-white relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-gradient-to-br from-purple-900/20 via-black to-blue-900/20" />
        <div className="absolute top-0 left-0 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse" />
        <div className="absolute bottom-0 right-0 w-96 h-96 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse" />
      </div>

      <div className="relative z-10 p-8">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 via-blue-500 to-cyan-500">
            🤖 ML Training Command Center
          </h1>
          <div className="flex gap-3">
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`px-4 py-2 rounded-xl backdrop-blur-xl border transition-all duration-300 ${
                autoRefresh 
                  ? 'bg-gradient-to-r from-green-500/20 to-emerald-500/20 border-green-500/50 text-green-400' 
                  : 'bg-gradient-to-r from-blue-500/20 to-indigo-500/20 border-blue-500/50 text-blue-400'
              }`}
            >
              <div className="flex items-center gap-2">
                <span>{autoRefresh ? '⏸️' : '▶️'}</span>
                <span>{autoRefresh ? 'Auto' : 'Manual'}</span>
              </div>
            </button>
            <button
              onClick={() => { refetchTraining(); refetchStats(); }}
              className="p-3 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 backdrop-blur-xl border border-cyan-500/50 rounded-xl text-cyan-400 hover:bg-cyan-500/30 transition-all duration-300 hover:scale-105"
            >
              🔄
            </button>
          </div>
        </div>

        {/* Overall Progress */}
        <div className="relative mb-8">
          <div className="absolute inset-0 bg-gradient-to-r from-green-500/20 to-emerald-500/20 rounded-2xl blur-xl" />
          <div className="relative bg-black/50 backdrop-blur-xl border border-green-500/30 rounded-2xl p-6">
            <h2 className="text-3xl font-bold text-green-400 mb-4">🚀 Overall Training Progress: All 4,190 Models</h2>
            {trainingSession && (
              <div className="space-y-4">
                <div>
                  <p className="text-xl text-gray-300 mb-2">
                    System Progress: {trainingSession.overall_progress}% Complete
                  </p>
                  <div className="relative">
                    <div className="w-full bg-gray-700 rounded-full h-6">
                      <div 
                        className="bg-gradient-to-r from-green-400 to-emerald-500 h-6 rounded-full transition-all duration-500"
                        style={{ width: `${trainingSession.overall_progress}%` }}
                      ></div>
                    </div>
                    <span className="absolute inset-0 flex items-center justify-center text-sm font-semibold text-white">
                      {trainingSession.overall_progress}%
                    </span>
                  </div>
                  <p className="text-lg text-gray-300 mt-2">
                    <span className="font-bold text-white">{trainingSession.completed_items}</span> of <span className="font-bold text-white">{trainingSession.total_queue_items}</span> models trained
                    ({trainingSession.remaining_models} remaining)
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Current Training & Controls */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Current Model Training */}
          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-2xl blur-xl" />
            <div className="relative bg-black/50 backdrop-blur-xl border border-blue-500/30 rounded-2xl p-6">
              <h3 className="text-2xl font-bold text-blue-400 mb-4">Current Model Training</h3>
              {trainingSession && (
                <div className="space-y-4">
                  <div>
                    <p className="text-gray-400 mb-2">
                      Training: {trainingSession.current_coin} - {trainingSession.current_model}
                    </p>
                    <div className="relative">
                      <div className="w-full bg-gray-700 rounded-full h-3">
                        <div 
                          className="bg-gradient-to-r from-blue-400 to-purple-500 h-3 rounded-full transition-all duration-500"
                          style={{ width: `${trainingSession.progress}%` }}
                        ></div>
                      </div>
                    </div>
                    <p className="text-sm text-gray-400 mt-1">
                      {trainingSession.progress}% - ETA: {formatTime(trainingSession.eta_seconds)}
                    </p>
                  </div>
                  
                  <div className="space-y-1 text-sm text-gray-400">
                    <p>Session started: {new Date(trainingSession.session_start_time).toLocaleString()}</p>
                    <p>Full completion ETA: {formatEstimatedCompletion(trainingSession.estimated_completion_time)}</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Training Controls */}
          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-orange-500/20 to-red-500/20 rounded-2xl blur-xl" />
            <div className="relative bg-black/50 backdrop-blur-xl border border-orange-500/30 rounded-2xl p-6">
              <h3 className="text-2xl font-bold text-orange-400 mb-4">Training Controls</h3>
              <div className="flex flex-wrap gap-3">
                <button
                  onClick={() => pauseTrainingMutation.mutate()}
                  disabled={pauseTrainingMutation.isPending}
                  className="px-4 py-2 bg-gradient-to-r from-yellow-500/20 to-orange-500/20 backdrop-blur-xl border border-yellow-500/50 rounded-xl text-yellow-400 hover:bg-yellow-500/30 transition-all duration-300 disabled:opacity-50"
                >
                  <div className="flex items-center gap-2">
                    <span>⏸️</span>
                    <span>Pause Training</span>
                  </div>
                </button>
                <button
                  onClick={() => resumeTrainingMutation.mutate()}
                  disabled={resumeTrainingMutation.isPending}
                  className="px-4 py-2 bg-gradient-to-r from-green-500/20 to-emerald-500/20 backdrop-blur-xl border border-green-500/50 rounded-xl text-green-400 hover:bg-green-500/30 transition-all duration-300 disabled:opacity-50"
                >
                  <div className="flex items-center gap-2">
                    <span>▶️</span>
                    <span>Resume Training</span>
                  </div>
                </button>
                <button
                  onClick={() => setSettingsOpen(true)}
                  className="px-4 py-2 bg-gradient-to-r from-purple-500/20 to-pink-500/20 backdrop-blur-xl border border-purple-500/50 rounded-xl text-purple-400 hover:bg-purple-500/30 transition-all duration-300"
                >
                  <div className="flex items-center gap-2">
                    <span>⚙️</span>
                    <span>Settings</span>
                  </div>
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Training Queue */}
        <div className="relative">
          <div className="absolute inset-0 bg-gradient-to-r from-gray-500/10 to-gray-500/10 rounded-2xl blur-xl" />
          <div className="relative bg-black/50 backdrop-blur-xl border border-gray-500/30 rounded-2xl overflow-hidden">
            <div className="p-6 border-b border-gray-700">
              <h3 className="text-2xl font-bold text-gray-100">
                Training Queue ({trainingQueue.length} items)
              </h3>
            </div>
            
            <div className="p-6">
              {queueLoading ? (
                <div className="text-center py-8">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-cyan-400 mx-auto mb-4"></div>
                  <p className="text-cyan-400">Loading training queue...</p>
                </div>
              ) : trainingQueue.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-white">
                    <thead>
                      <tr className="border-b border-gray-700">
                        <th className="text-left p-4 text-cyan-400 font-bold">Position</th>
                        <th className="text-left p-4 text-cyan-400 font-bold">Coin</th>
                        <th className="text-left p-4 text-cyan-400 font-bold">Model Type</th>
                        <th className="text-left p-4 text-cyan-400 font-bold">Status</th>
                        <th className="text-left p-4 text-cyan-400 font-bold">Progress</th>
                        <th className="text-left p-4 text-cyan-400 font-bold">Started</th>
                      </tr>
                    </thead>
                    <tbody>
                      {trainingQueue.slice(0, 20).map((item: any, index: number) => (
                        <tr key={`${item.coin_symbol}-${item.model_type}`} className="border-b border-gray-800 hover:bg-gradient-to-r hover:from-purple-500/10 hover:to-transparent transition-all duration-300">
                          <td className="p-4">
                            <span className="font-bold text-white">#{item.queue_position}</span>
                          </td>
                          <td className="p-4">
                            <span className="font-bold text-white">{item.coin_symbol}</span>
                          </td>
                          <td className="p-4">
                            <div className="inline-flex items-center px-3 py-1 rounded-lg bg-gradient-to-r from-blue-500/20 to-indigo-500/20 border border-blue-500/50">
                              <span className="text-blue-400 font-semibold text-sm">{item.model_type}</span>
                            </div>
                          </td>
                          <td className="p-4">
                            <div className={`inline-flex items-center px-3 py-1 rounded-lg ${
                              item.status === 'completed' ? 'bg-gradient-to-r from-green-500/20 to-emerald-500/20 border border-green-500/50' :
                              item.status === 'training' ? 'bg-gradient-to-r from-yellow-500/20 to-orange-500/20 border border-yellow-500/50' :
                              item.status === 'error' ? 'bg-gradient-to-r from-red-500/20 to-pink-500/20 border border-red-500/50' :
                              'bg-gradient-to-r from-gray-500/20 to-gray-500/20 border border-gray-500/50'
                            }`}>
                              <span className={`font-semibold text-sm ${
                                item.status === 'completed' ? 'text-green-400' :
                                item.status === 'training' ? 'text-yellow-400' :
                                item.status === 'error' ? 'text-red-400' :
                                'text-gray-400'
                              }`}>
                                {item.status}
                              </span>
                            </div>
                          </td>
                          <td className="p-4">
                            <div className="w-24">
                              <div className="w-full bg-gray-700 rounded-full h-2">
                                <div 
                                  className="bg-gradient-to-r from-green-400 to-emerald-500 h-2 rounded-full transition-all duration-500"
                                  style={{ width: `${item.progress}%` }}
                                ></div>
                              </div>
                              <span className="text-xs text-gray-400 mt-1">{item.progress}%</span>
                            </div>
                          </td>
                          <td className="p-4 text-gray-400 text-sm">
                            {item.started_at ? new Date(item.started_at).toLocaleTimeString() : '-'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-center py-8">
                  <div className="w-16 h-16 bg-gradient-to-br from-blue-400 to-indigo-600 rounded-xl flex items-center justify-center mx-auto mb-4">
                    <span className="text-3xl">📋</span>
                  </div>
                  <h3 className="text-xl font-bold text-blue-400 mb-2">No Training Items</h3>
                  <p className="text-gray-300">No training items currently in queue</p>
                </div>
              )}
              
              {trainingQueue.length > 20 && (
                <p className="text-sm text-gray-400 mt-4">
                  Showing first 20 items. {trainingQueue.length - 20} more items in queue.
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Training Settings Modal */}
        {settingsOpen && (
          <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="relative max-w-md w-full">
              <div className="absolute inset-0 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-2xl blur-xl" />
              <div className="relative bg-black/90 backdrop-blur-xl border border-purple-500/30 rounded-2xl p-6">
                <h3 className="text-2xl font-bold text-purple-400 mb-6">Training Settings</h3>
                
                <div className="space-y-6">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Auto-retrain models</span>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        checked={settings.autoRetrain}
                        onChange={(e) => setSettings({...settings, autoRetrain: e.target.checked})}
                        className="sr-only peer"
                      />
                      <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-purple-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
                    </label>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm mb-2">Training Interval (minutes): {settings.trainingInterval / 60}</label>
                    <input
                      type="range"
                      min="5"
                      max="720"
                      step="5"
                      value={settings.trainingInterval / 60}
                      onChange={(e) => setSettings({...settings, trainingInterval: Number(e.target.value) * 60})}
                      className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
                    />
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm mb-2">Max Models per Coin: {settings.maxModelsPerCoin}</label>
                    <input
                      type="range"
                      min="1"
                      max="10"
                      step="1"
                      value={settings.maxModelsPerCoin}
                      onChange={(e) => setSettings({...settings, maxModelsPerCoin: Number(e.target.value)})}
                      className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
                    />
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm mb-2">Training Batch Size: {settings.batchSize}</label>
                    <input
                      type="range"
                      min="1"
                      max="20"
                      step="1"
                      value={settings.batchSize}
                      onChange={(e) => setSettings({...settings, batchSize: Number(e.target.value)})}
                      className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Enable notifications</span>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        checked={settings.enableNotifications}
                        onChange={(e) => setSettings({...settings, enableNotifications: e.target.checked})}
                        className="sr-only peer"
                      />
                      <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-purple-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
                    </label>
                  </div>
                </div>

                <div className="flex gap-3 mt-8">
                  <button
                    onClick={() => setSettingsOpen(false)}
                    className="flex-1 px-4 py-2 bg-gradient-to-r from-gray-500/20 to-gray-500/20 backdrop-blur-xl border border-gray-500/50 rounded-xl text-gray-400 hover:bg-gray-500/30 transition-all duration-300"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={() => {
                      toast.success('Settings saved successfully');
                      setSettingsOpen(false);
                    }}
                    className="flex-1 px-4 py-2 bg-gradient-to-r from-purple-500/20 to-pink-500/20 backdrop-blur-xl border border-purple-500/50 rounded-xl text-purple-400 hover:bg-purple-500/30 transition-all duration-300"
                  >
                    Save Settings
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      <style jsx>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: linear-gradient(45deg, #a855f7, #ec4899);
          cursor: pointer;
          border: 2px solid #1f2937;
        }
        .slider::-moz-range-thumb {
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: linear-gradient(45deg, #a855f7, #ec4899);
          cursor: pointer;
          border: 2px solid #1f2937;
        }
      `}</style>
    </div>
  );
};