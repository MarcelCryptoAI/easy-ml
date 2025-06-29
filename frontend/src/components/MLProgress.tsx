import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Alert,
  IconButton
} from '@mui/material';
import { Refresh } from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { tradingApi } from '../utils/api';

interface MLTrainingStatus {
  coin_symbol: string;
  last_trained: string;
  models_trained: string[];
  training_status: string;
  accuracy_scores: { [key: string]: number };
  confidence_scores: { [key: string]: number };
  current_predictions: { [key: string]: string };
}

export const MLProgress: React.FC = () => {
  const { data: coins = [], refetch: refetchCoins, error: coinsError } = useQuery({
    queryKey: ['coins'],
    queryFn: tradingApi.getCoins,
    refetchInterval: 30000,
    retry: 3,
    retryDelay: 1000
  });

  const { data: mlStatus = [], isLoading, refetch } = useQuery({
    queryKey: ['ml-training-status'],
    queryFn: async () => {
      // Get ML training status for all coins (remove limit)
      const statusPromises = coins.map(async (coin) => {
        try {
          const predictions = await tradingApi.getPredictions(coin.symbol);
          return {
            coin_symbol: coin.symbol,
            last_trained: predictions[0]?.created_at || null,
            models_trained: predictions.map(p => p.model_type),
            training_status: predictions.length === 10 ? 'complete' : 'training',
            accuracy_scores: predictions.reduce((acc, p) => {
              acc[p.model_type] = p.confidence;
              return acc;
            }, {} as { [key: string]: number }),
            confidence_scores: predictions.reduce((acc, p) => {
              acc[p.model_type] = p.confidence;
              return acc;
            }, {} as { [key: string]: number }),
            current_predictions: predictions.reduce((acc, p) => {
              acc[p.model_type] = p.prediction;
              return acc;
            }, {} as { [key: string]: string })
          };
        } catch (error) {
          return {
            coin_symbol: coin.symbol,
            last_trained: null,
            models_trained: [],
            training_status: 'pending',
            accuracy_scores: {},
            confidence_scores: {},
            current_predictions: {}
          };
        }
      });
      
      return Promise.all(statusPromises);
    },
    enabled: coins.length > 0,
    refetchInterval: 15000
  });

  const totalCoins = coins.length;
  const completedCoins = mlStatus.filter(status => status.training_status === 'complete').length;
  const trainingCoins = mlStatus.filter(status => status.training_status === 'training').length;
  const pendingCoins = mlStatus.filter(status => status.training_status === 'pending').length;

  const overallProgress = totalCoins > 0 ? (completedCoins / totalCoins) * 100 : 0;

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'complete': return 'success';
      case 'training': return 'warning';
      case 'pending': return 'default';
      default: return 'default';
    }
  };

  const getModelTypeColor = (modelType: string) => {
    switch (modelType) {
      case 'lstm': return 'primary';
      case 'random_forest': return 'secondary';
      case 'svm': return 'info';
      case 'neural_network': return 'success';
      case 'xgboost': return 'warning';
      case 'lightgbm': return 'error';
      case 'catboost': return 'primary';
      case 'transformer': return 'secondary';
      case 'gru': return 'info';
      case 'cnn_1d': return 'success';
      default: return 'default';
    }
  };

  return (
    <div className="min-h-screen bg-black text-white relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-gradient-to-br from-purple-900/20 via-black to-blue-900/20" />
        <div className="absolute top-0 left-0 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse" />
        <div className="absolute bottom-0 right-0 w-96 h-96 bg-cyan-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse" />
      </div>

      <div className="relative z-10 p-8">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-purple-500 to-pink-500">
            📊 ML Training Progress
          </h1>
          <button
            onClick={() => refetch()}
            disabled={isLoading}
            className="px-6 py-3 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 backdrop-blur-xl border border-cyan-500/50 rounded-xl text-cyan-400 hover:bg-cyan-500/30 transition-all duration-300 hover:scale-105 hover:shadow-[0_0_30px_rgba(0,255,255,0.5)] disabled:opacity-50"
          >
            <Refresh className="w-6 h-6" />
          </button>
        </div>

        {/* Enhanced Statistics */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-6 mb-8">
          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 rounded-2xl blur-xl" />
            <div className="relative bg-black/50 backdrop-blur-xl border border-cyan-500/30 rounded-2xl p-6">
              <p className="text-gray-400 text-sm mb-2">Total Coins</p>
              <p className="text-4xl font-bold text-cyan-400">{totalCoins}</p>
              <p className="text-gray-500 text-sm mt-2">Active trading pairs</p>
            </div>
          </div>

          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-2xl blur-xl" />
            <div className="relative bg-black/50 backdrop-blur-xl border border-purple-500/30 rounded-2xl p-6">
              <p className="text-gray-400 text-sm mb-2">Overall Progress</p>
              <p className="text-4xl font-bold text-purple-400">{overallProgress.toFixed(1)}%</p>
              <div className="mt-3 h-2 bg-gray-800 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-purple-400 to-pink-400 rounded-full transition-all duration-1000"
                  style={{ width: `${overallProgress}%` }}
                />
              </div>
            </div>
          </div>

          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-green-500/20 to-emerald-500/20 rounded-2xl blur-xl" />
            <div className="relative bg-black/50 backdrop-blur-xl border border-green-500/30 rounded-2xl p-6">
              <p className="text-gray-400 text-sm mb-2">Complete (10/10)</p>
              <p className="text-4xl font-bold text-green-400">{completedCoins}</p>
              <p className="text-gray-500 text-sm mt-2">All models trained</p>
            </div>
          </div>

          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-yellow-500/20 to-orange-500/20 rounded-2xl blur-xl" />
            <div className="relative bg-black/50 backdrop-blur-xl border border-yellow-500/30 rounded-2xl p-6">
              <p className="text-gray-400 text-sm mb-2">Training</p>
              <p className="text-4xl font-bold text-yellow-400">{trainingCoins}</p>
              <p className="text-gray-500 text-sm mt-2">In progress</p>
            </div>
          </div>

          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-gray-500/20 to-gray-500/20 rounded-2xl blur-xl" />
            <div className="relative bg-black/50 backdrop-blur-xl border border-gray-500/30 rounded-2xl p-6">
              <p className="text-gray-400 text-sm mb-2">Pending (0/10)</p>
              <p className="text-4xl font-bold text-gray-400">{pendingCoins}</p>
              <p className="text-gray-500 text-sm mt-2">Not started</p>
            </div>
          </div>
        </div>

        {/* Connection Status */}
        {coinsError && (
          <div className="relative mb-6">
            <div className="absolute inset-0 bg-gradient-to-r from-red-500/20 to-pink-500/20 rounded-2xl blur-xl" />
            <div className="relative bg-black/50 backdrop-blur-xl border border-red-500/30 rounded-2xl p-6">
              <div className="flex items-center gap-3">
                <span className="text-2xl">❌</span>
                <div>
                  <p className="text-red-400 font-bold">Backend connection failed. Check if backend is running.</p>
                  <p className="text-gray-400 text-sm">Backend URL: {process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Detailed Training Status */}
        <div className="relative">
          <div className="absolute inset-0 bg-gradient-to-r from-gray-500/10 to-gray-500/10 rounded-2xl blur-xl" />
          <div className="relative bg-black/50 backdrop-blur-xl border border-gray-500/30 rounded-2xl p-6">
            <h2 className="text-2xl font-bold text-white mb-6">
              Detailed Training Status (All {mlStatus.length} Coins)
            </h2>
            
            {isLoading ? (
              <div className="text-center py-8">
                <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-cyan-400"></div>
                <p className="text-gray-400 mt-4">Loading ML training status...</p>
              </div>
            ) : coinsError ? (
              <div className="text-center py-8">
                <p className="text-red-400">Failed to load training status. Backend may be disconnected.</p>
              </div>
            ) : mlStatus.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-white">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left p-4 text-cyan-400 font-bold">Coin</th>
                      <th className="text-left p-4 text-cyan-400 font-bold">Status</th>
                      <th className="text-left p-4 text-cyan-400 font-bold">Models Trained</th>
                      <th className="text-left p-4 text-cyan-400 font-bold">Last Trained</th>
                      <th className="text-left p-4 text-cyan-400 font-bold">Avg Confidence</th>
                      <th className="text-left p-4 text-cyan-400 font-bold">Current Predictions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {mlStatus.map((status) => {
                      const avgConfidence = Object.values(status.confidence_scores).length > 0
                        ? Object.values(status.confidence_scores).reduce((a, b) => a + b, 0) / Object.values(status.confidence_scores).length
                        : 0;

                      const getStatusStyle = (status: string) => {
                        switch (status) {
                          case 'complete': return 'bg-green-500/20 border-green-500/50 text-green-400';
                          case 'training': return 'bg-yellow-500/20 border-yellow-500/50 text-yellow-400';
                          case 'pending': return 'bg-gray-500/20 border-gray-500/50 text-gray-400';
                          default: return 'bg-gray-500/20 border-gray-500/50 text-gray-400';
                        }
                      };

                      return (
                        <tr key={status.coin_symbol} className="border-b border-gray-800 hover:bg-gradient-to-r hover:from-purple-500/10 hover:to-transparent transition-all duration-300">
                          <td className="p-4">
                            <span className="font-bold text-white">{status.coin_symbol}</span>
                          </td>
                          
                          <td className="p-4">
                            <span className={`inline-flex px-3 py-1 rounded-lg border ${getStatusStyle(status.training_status)}`}>
                              {status.training_status}
                            </span>
                          </td>
                          
                          <td className="p-4">
                            <div className="flex gap-1 flex-wrap">
                              {['lstm', 'random_forest', 'svm', 'neural_network', 'xgboost', 'lightgbm', 'catboost', 'transformer', 'gru', 'cnn_1d'].map((modelType) => {
                                const isTrained = status.models_trained.includes(modelType);
                                return (
                                  <span
                                    key={modelType}
                                    className={`inline-flex px-2 py-1 rounded text-xs font-medium ${
                                      isTrained 
                                        ? 'bg-cyan-500/20 border border-cyan-500/50 text-cyan-400' 
                                        : 'bg-gray-700/20 border border-gray-700/50 text-gray-500'
                                    }`}
                                  >
                                    {modelType.replace('_', ' ').toUpperCase()}
                                  </span>
                                );
                              })}
                            </div>
                            <div className="flex items-center gap-2 mt-2">
                              <span className={`inline-flex px-2 py-1 rounded text-xs font-medium ${
                                status.models_trained.length === 10 
                                  ? 'bg-green-500/20 border border-green-500/50 text-green-400'
                                  : status.models_trained.length >= 5 
                                  ? 'bg-yellow-500/20 border border-yellow-500/50 text-yellow-400'
                                  : 'bg-red-500/20 border border-red-500/50 text-red-400'
                              }`}>
                                {status.models_trained.length}/10
                              </span>
                              <div className="w-16 h-2 bg-gray-700 rounded-full overflow-hidden">
                                <div 
                                  className={`h-full rounded-full transition-all duration-500 ${
                                    status.models_trained.length === 10 
                                      ? 'bg-green-400' 
                                      : 'bg-cyan-400'
                                  }`}
                                  style={{ width: `${(status.models_trained.length / 10) * 100}%` }}
                                />
                              </div>
                              <span className="text-gray-500 text-xs">
                                {((status.models_trained.length / 10) * 100).toFixed(0)}%
                              </span>
                            </div>
                          </td>
                          
                          <td className="p-4 text-gray-300">
                            {status.last_trained 
                              ? new Date(status.last_trained).toLocaleString()
                              : 'Not trained'
                            }
                          </td>
                          
                          <td className="p-4">
                            {avgConfidence > 0 ? (
                              <span className={`inline-flex px-3 py-1 rounded-lg border ${
                                avgConfidence >= 70 
                                  ? 'bg-green-500/20 border-green-500/50 text-green-400'
                                  : avgConfidence >= 50 
                                  ? 'bg-yellow-500/20 border-yellow-500/50 text-yellow-400'
                                  : 'bg-red-500/20 border-red-500/50 text-red-400'
                              }`}>
                                {avgConfidence.toFixed(1)}%
                              </span>
                            ) : (
                              <span className="text-gray-500">-</span>
                            )}
                          </td>
                          
                          <td className="p-4">
                            <div className="flex gap-1 flex-wrap">
                              {Object.entries(status.current_predictions).map(([model, prediction]) => (
                                <span
                                  key={model}
                                  className={`inline-flex px-2 py-1 rounded text-xs border ${
                                    prediction === 'buy' || prediction === 'LONG'
                                      ? 'bg-green-500/20 border-green-500/50 text-green-400' 
                                      : prediction === 'sell' || prediction === 'SHORT'
                                      ? 'bg-red-500/20 border-red-500/50 text-red-400'
                                      : 'bg-gray-700/20 border-gray-700/50 text-gray-500'
                                  }`}
                                >
                                  {model}: {prediction}
                                </span>
                              ))}
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-center py-8">
                <p className="text-blue-400">ML training data will appear here as models complete training.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};