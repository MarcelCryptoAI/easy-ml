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
  const { data: coins = [], refetch: refetchCoins } = useQuery({
    queryKey: ['coins'],
    queryFn: tradingApi.getCoins,
    refetchInterval: 30000
  });

  const { data: mlStatus = [], isLoading, refetch } = useQuery({
    queryKey: ['ml-training-status'],
    queryFn: async () => {
      // Get ML training status for all coins
      const statusPromises = coins.slice(0, 20).map(async (coin) => {
        try {
          const predictions = await tradingApi.getPredictions(coin.symbol);
          return {
            coin_symbol: coin.symbol,
            last_trained: predictions[0]?.created_at || null,
            models_trained: predictions.map(p => p.model_type),
            training_status: predictions.length === 4 ? 'complete' : 'training',
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
      default: return 'default';
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">ML Training Progress</Typography>
        <IconButton onClick={() => refetch()} disabled={isLoading}>
          <Refresh />
        </IconButton>
      </Box>

      {/* Overall Progress */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Overall Progress
              </Typography>
              <Typography variant="h4" color="primary">
                {overallProgress.toFixed(1)}%
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={overallProgress} 
                sx={{ mt: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Completed
              </Typography>
              <Typography variant="h4" color="success.main">
                {completedCoins}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                of {totalCoins} coins
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Training
              </Typography>
              <Typography variant="h4" color="warning.main">
                {trainingCoins}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                actively training
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Pending
              </Typography>
              <Typography variant="h4" color="text.secondary">
                {pendingCoins}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                waiting in queue
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Detailed Training Status */}
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Detailed Training Status (Top 20 Coins)
        </Typography>
        
        {isLoading ? (
          <Alert severity="info">Loading ML training status...</Alert>
        ) : mlStatus.length > 0 ? (
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Coin</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Models Trained</TableCell>
                  <TableCell>Last Trained</TableCell>
                  <TableCell>Avg Confidence</TableCell>
                  <TableCell>Current Predictions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {mlStatus.map((status) => {
                  const avgConfidence = Object.values(status.confidence_scores).length > 0
                    ? Object.values(status.confidence_scores).reduce((a, b) => a + b, 0) / Object.values(status.confidence_scores).length
                    : 0;

                  return (
                    <TableRow key={status.coin_symbol}>
                      <TableCell>
                        <Typography variant="subtitle2" fontWeight="bold">
                          {status.coin_symbol}
                        </Typography>
                      </TableCell>
                      
                      <TableCell>
                        <Chip 
                          label={status.training_status}
                          color={getStatusColor(status.training_status)}
                          size="small"
                        />
                      </TableCell>
                      
                      <TableCell>
                        <Box display="flex" gap={0.5} flexWrap="wrap">
                          {status.models_trained.map((model) => (
                            <Chip
                              key={model}
                              label={model}
                              color={getModelTypeColor(model)}
                              size="small"
                              variant="outlined"
                            />
                          ))}
                        </Box>
                      </TableCell>
                      
                      <TableCell>
                        {status.last_trained 
                          ? new Date(status.last_trained).toLocaleString()
                          : 'Not trained'
                        }
                      </TableCell>
                      
                      <TableCell>
                        {avgConfidence > 0 ? (
                          <Chip
                            label={`${avgConfidence.toFixed(1)}%`}
                            color={avgConfidence >= 70 ? 'success' : avgConfidence >= 50 ? 'warning' : 'error'}
                            size="small"
                          />
                        ) : (
                          '-'
                        )}
                      </TableCell>
                      
                      <TableCell>
                        <Box display="flex" gap={0.5} flexWrap="wrap">
                          {Object.entries(status.current_predictions).map(([model, prediction]) => (
                            <Chip
                              key={model}
                              label={`${model}: ${prediction}`}
                              color={prediction === 'buy' ? 'success' : prediction === 'sell' ? 'error' : 'default'}
                              size="small"
                              variant="outlined"
                            />
                          ))}
                        </Box>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        ) : (
          <Alert severity="info">
            ML training data will appear here as models complete training.
          </Alert>
        )}
      </Paper>
    </Box>
  );
};