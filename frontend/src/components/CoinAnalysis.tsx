import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  CircularProgress,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import { useQuery } from '@tanstack/react-query';
import { tradingApi } from '../utils/api';
import { Coin, MLPrediction, TradingStrategy } from '../types';
import toast from 'react-hot-toast';

export const CoinAnalysis: React.FC = () => {
  const [selectedCoin, setSelectedCoin] = useState<string>('');
  const [optimizing, setOptimizing] = useState(false);

  const { data: coins = [] } = useQuery({
    queryKey: ['coins'],
    queryFn: tradingApi.getCoins
  });

  const { data: predictions = [], isLoading: predictionsLoading } = useQuery({
    queryKey: ['predictions', selectedCoin],
    queryFn: () => tradingApi.getPredictions(selectedCoin),
    enabled: !!selectedCoin,
    refetchInterval: 30000
  });

  const { data: strategy, refetch: refetchStrategy } = useQuery({
    queryKey: ['strategy', selectedCoin],
    queryFn: () => tradingApi.getStrategy(selectedCoin),
    enabled: !!selectedCoin
  });

  const handleOptimizeStrategy = async () => {
    if (!selectedCoin) return;
    
    setOptimizing(true);
    try {
      const result = await tradingApi.optimizeStrategy(selectedCoin);
      if (result.success) {
        toast.success('Strategy optimized successfully');
        refetchStrategy();
      } else {
        toast.error(result.error || 'Optimization failed');
      }
    } catch (error) {
      toast.error('Failed to optimize strategy');
    } finally {
      setOptimizing(false);
    }
  };

  const getSignalColor = (prediction: string) => {
    switch (prediction) {
      case 'buy': return 'success';
      case 'sell': return 'error';
      default: return 'default';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return 'success';
    if (confidence >= 60) return 'warning';
    return 'error';
  };

  const averageConfidence = predictions.length > 0 
    ? predictions.reduce((sum, pred) => sum + pred.confidence, 0) / predictions.length
    : 0;

  const buySignals = predictions.filter(p => p.prediction === 'buy').length;
  const sellSignals = predictions.filter(p => p.prediction === 'sell').length;
  const holdSignals = predictions.filter(p => p.prediction === 'hold').length;

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Coin Analysis
      </Typography>

      <Grid container spacing={3}>
        {/* Coin Selection */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <FormControl fullWidth>
              <InputLabel>Select Coin</InputLabel>
              <Select
                value={selectedCoin}
                onChange={(e) => setSelectedCoin(e.target.value)}
                label="Select Coin"
              >
                {coins.map((coin: Coin) => (
                  <MenuItem key={coin.id} value={coin.symbol}>
                    {coin.symbol} - {coin.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Paper>
        </Grid>

        {selectedCoin && (
          <>
            {/* ML Predictions Summary */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2 }}>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                  <Typography variant="h6">
                    ML Predictions for {selectedCoin}
                  </Typography>
                  {predictionsLoading && <CircularProgress size={20} />}
                </Box>

                <Grid container spacing={2} mb={2}>
                  <Grid item xs={4}>
                    <Card>
                      <CardContent sx={{ textAlign: 'center' }}>
                        <Typography color="textSecondary">Buy Signals</Typography>
                        <Typography variant="h4" color="success.main">
                          {buySignals}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={4}>
                    <Card>
                      <CardContent sx={{ textAlign: 'center' }}>
                        <Typography color="textSecondary">Hold Signals</Typography>
                        <Typography variant="h4">
                          {holdSignals}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={4}>
                    <Card>
                      <CardContent sx={{ textAlign: 'center' }}>
                        <Typography color="textSecondary">Sell Signals</Typography>
                        <Typography variant="h4" color="error.main">
                          {sellSignals}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>

                <Card sx={{ mb: 2 }}>
                  <CardContent>
                    <Typography color="textSecondary">Average Confidence</Typography>
                    <Typography variant="h5">
                      <Chip 
                        label={`${averageConfidence.toFixed(1)}%`}
                        color={getConfidenceColor(averageConfidence)}
                      />
                    </Typography>
                  </CardContent>
                </Card>

                {/* Individual Predictions */}
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Model</TableCell>
                        <TableCell>Prediction</TableCell>
                        <TableCell>Confidence</TableCell>
                        <TableCell>Time</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {predictions.map((prediction: MLPrediction) => (
                        <TableRow key={prediction.model_type}>
                          <TableCell>{prediction.model_type}</TableCell>
                          <TableCell>
                            <Chip 
                              label={prediction.prediction}
                              color={getSignalColor(prediction.prediction)}
                              size="small"
                            />
                          </TableCell>
                          <TableCell>
                            <Chip 
                              label={`${prediction.confidence.toFixed(1)}%`}
                              color={getConfidenceColor(prediction.confidence)}
                              size="small"
                            />
                          </TableCell>
                          <TableCell>
                            {new Date(prediction.created_at).toLocaleTimeString()}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Paper>
            </Grid>

            {/* Trading Strategy */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2 }}>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                  <Typography variant="h6">
                    Trading Strategy
                  </Typography>
                  <Button
                    variant="contained"
                    onClick={handleOptimizeStrategy}
                    disabled={optimizing}
                    color="secondary"
                  >
                    {optimizing ? <CircularProgress size={20} /> : 'AI Optimize'}
                  </Button>
                </Box>

                {strategy && (
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Card>
                        <CardContent>
                          <Typography color="textSecondary" variant="body2">
                            Take Profit
                          </Typography>
                          <Typography variant="h6">
                            {strategy.take_profit_percentage}%
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                    <Grid item xs={6}>
                      <Card>
                        <CardContent>
                          <Typography color="textSecondary" variant="body2">
                            Stop Loss
                          </Typography>
                          <Typography variant="h6">
                            {strategy.stop_loss_percentage}%
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                    <Grid item xs={6}>
                      <Card>
                        <CardContent>
                          <Typography color="textSecondary" variant="body2">
                            Leverage
                          </Typography>
                          <Typography variant="h6">
                            {strategy.leverage}x
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                    <Grid item xs={6}>
                      <Card>
                        <CardContent>
                          <Typography color="textSecondary" variant="body2">
                            Confidence Threshold
                          </Typography>
                          <Typography variant="h6">
                            {strategy.confidence_threshold}%
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                    
                    {strategy.updated_by_ai && (
                      <Grid item xs={12}>
                        <Card sx={{ bgcolor: 'info.light' }}>
                          <CardContent>
                            <Typography variant="subtitle2" gutterBottom>
                              🤖 AI Optimization Applied
                            </Typography>
                            <Typography variant="body2">
                              {strategy.ai_optimization_reason}
                            </Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                    )}
                  </Grid>
                )}
              </Paper>
            </Grid>
          </>
        )}
      </Grid>
    </Box>
  );
};