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
  TableRow,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  RadioGroup,
  FormControlLabel,
  Radio,
  Divider,
  IconButton,
  Tooltip
} from '@mui/material';
import { 
  TrendingUp, 
  TrendingDown, 
  Speed, 
  PlayArrow,
  Warning 
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { tradingApi } from '../utils/api';
import { Coin, MLPrediction, TradingStrategy } from '../types';
import toast from 'react-hot-toast';

interface ManualTradeData {
  amount_percentage: number;
  order_type: 'market' | 'limit';
  limit_price?: number;
  leverage: number;
  margin_mode: 'cross' | 'isolated';
  take_profit_percentage: number;
  stop_loss_percentage: number;
  side: 'buy' | 'sell';
}

export const CoinAnalysis: React.FC = () => {
  const [selectedCoin, setSelectedCoin] = useState<string>('');
  const [optimizing, setOptimizing] = useState(false);
  const [priorityTraining, setPriorityTraining] = useState(false);
  const [tradeDialogOpen, setTradeDialogOpen] = useState(false);
  const [confirmDialogOpen, setConfirmDialogOpen] = useState(false);
  const [currentPrice, setCurrentPrice] = useState(0);
  const [availableBalance, setAvailableBalance] = useState(0);
  const [tradeData, setTradeData] = useState<ManualTradeData>({
    amount_percentage: 5,
    order_type: 'market',
    leverage: 10,
    margin_mode: 'cross',
    take_profit_percentage: 2.5,
    stop_loss_percentage: 1.5,
    side: 'buy'
  });
  const queryClient = useQueryClient();

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

  // Priority training mutation
  const priorityTrainingMutation = useMutation({
    mutationFn: async (coinSymbol: string) => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/training/priority`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ coin_symbol: coinSymbol })
      });
      if (!response.ok) throw new Error('Failed to start priority training');
      return response.json();
    },
    onSuccess: () => {
      toast.success('🚀 Priority training started for all 10 models!');
      setPriorityTraining(true);
      // Auto-refresh predictions
      setTimeout(() => {
        queryClient.invalidateQueries({ queryKey: ['predictions', selectedCoin] });
      }, 2000);
    },
    onError: () => {
      toast.error('Failed to start priority training');
    }
  });

  // Manual trade mutation
  const manualTradeMutation = useMutation({
    mutationFn: async (trade: ManualTradeData & { coin_symbol: string, current_price: number }) => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/trading/manual`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(trade)
      });
      if (!response.ok) throw new Error('Failed to execute manual trade');
      return response.json();
    },
    onSuccess: (data) => {
      toast.success(`✅ ${tradeData.side.toUpperCase()} order executed!`);
      setConfirmDialogOpen(false);
      setTradeDialogOpen(false);
      queryClient.invalidateQueries({ queryKey: ['trading-status'] });
    },
    onError: (error) => {
      toast.error(`❌ Trade failed: ${error.message}`);
    }
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

  const handlePriorityTraining = () => {
    if (!selectedCoin) return;
    priorityTrainingMutation.mutate(selectedCoin);
  };

  const handleOpenTradeDialog = async (side: 'buy' | 'sell') => {
    if (!selectedCoin) return;
    
    // Get current price and balance
    try {
      const [priceResponse, balanceResponse] = await Promise.all([
        fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/price/${selectedCoin}`),
        fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/trading/status`)
      ]);
      
      const priceData = await priceResponse.json();
      const balanceData = await balanceResponse.json();
      
      setCurrentPrice(priceData.price || 0);
      setAvailableBalance(balanceData.balance || 0);
      setTradeData(prev => ({ ...prev, side }));
      setTradeDialogOpen(true);
    } catch (error) {
      toast.error('Failed to get current price and balance');
    }
  };

  const handleConfirmTrade = () => {
    setConfirmDialogOpen(true);
  };

  const handleExecuteTrade = () => {
    if (!selectedCoin) return;
    
    manualTradeMutation.mutate({
      ...tradeData,
      coin_symbol: selectedCoin,
      current_price: currentPrice
    });
  };

  // Calculate leveraged percentages for display
  const calculateLeverageDisplay = (percentage: number, leverage: number, isProfit: boolean) => {
    const leveragedPercentage = percentage * leverage;
    const sign = isProfit ? '+' : '-';
    return `${percentage}% (${sign}${leveragedPercentage}%)`;
  };

  // Calculate trade amounts
  const calculateTradeAmounts = () => {
    const tradeAmount = (availableBalance * tradeData.amount_percentage) / 100;
    const positionSize = tradeAmount / currentPrice;
    return { tradeAmount, positionSize };
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

  // ALLEEN echte ML data - geen fallback waardes
  const averageConfidence = predictions?.length > 0 
    ? predictions.reduce((sum, pred) => sum + pred.confidence, 0) / predictions.length
    : null;

  const buySignals = predictions?.filter(p => p.prediction === 'buy').length || 0;
  const sellSignals = predictions?.filter(p => p.prediction === 'sell').length || 0;
  const holdSignals = predictions?.filter(p => p.prediction === 'hold').length || 0;

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
            
            {/* Action Buttons */}
            {selectedCoin && (
              <Box sx={{ mt: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                <Button
                  variant="contained"
                  color="secondary"
                  onClick={handlePriorityTraining}
                  disabled={priorityTrainingMutation.isPending || priorityTraining}
                  startIcon={<Speed />}
                >
                  {priorityTraining ? 'Training in Progress...' : 'Priority Train All 10 Models'}
                </Button>
                
                <Button
                  variant="contained"
                  color="success"
                  onClick={() => handleOpenTradeDialog('buy')}
                  startIcon={<TrendingUp />}
                >
                  Manual LONG
                </Button>
                
                <Button
                  variant="contained"
                  color="error"
                  onClick={() => handleOpenTradeDialog('sell')}
                  startIcon={<TrendingDown />}
                >
                  Manual SHORT
                </Button>
              </Box>
            )}
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
                      {averageConfidence !== null ? (
                        <Chip 
                          label={`${averageConfidence.toFixed(1)}%`}
                          color={getConfidenceColor(averageConfidence)}
                        />
                      ) : (
                        <Typography variant="body2" color="text.secondary">
                          Waiting for ML predictions...
                        </Typography>
                      )}
                    </Typography>
                  </CardContent>
                </Card>

                {/* Individual Predictions - ALLEEN echte ML data */}
                {predictions?.length > 0 ? (
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
                ) : (
                  <Alert severity="info">
                    ML models are training. Predictions will appear here once complete.
                  </Alert>
                )}
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

      {/* Manual Trading Dialog */}
      <Dialog open={tradeDialogOpen} onClose={() => setTradeDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          Manual {tradeData.side === 'buy' ? 'LONG' : 'SHORT'} Order - {selectedCoin}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2 }}>
            <Grid container spacing={3}>
              {/* Current Price & Balance */}
              <Grid item xs={12}>
                <Alert severity="info">
                  <strong>Current Price:</strong> ${currentPrice.toFixed(4)} | 
                  <strong> Available Balance:</strong> {availableBalance.toFixed(2)} USDT
                </Alert>
              </Grid>
              
              {/* Amount Percentage */}
              <Grid item xs={12} sm={6}>
                <TextField
                  label="Amount (% of balance)"
                  type="number"
                  value={tradeData.amount_percentage}
                  onChange={(e) => setTradeData(prev => ({ ...prev, amount_percentage: Number(e.target.value) }))}
                  inputProps={{ min: 1, max: 100, step: 1 }}
                  fullWidth
                  helperText={`≈ ${calculateTradeAmounts().tradeAmount.toFixed(2)} USDT`}
                />
              </Grid>

              {/* Order Type */}
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Order Type</InputLabel>
                  <Select
                    value={tradeData.order_type}
                    onChange={(e) => setTradeData(prev => ({ ...prev, order_type: e.target.value as 'market' | 'limit' }))}
                    label="Order Type"
                  >
                    <MenuItem value="market">Market Order</MenuItem>
                    <MenuItem value="limit">Limit Order</MenuItem>
                  </Select>
                </FormControl>
              </Grid>

              {/* Limit Price (if limit order) */}
              {tradeData.order_type === 'limit' && (
                <Grid item xs={12} sm={6}>
                  <TextField
                    label="Limit Price (USDT)"
                    type="number"
                    value={tradeData.limit_price || currentPrice}
                    onChange={(e) => setTradeData(prev => ({ ...prev, limit_price: Number(e.target.value) }))}
                    inputProps={{ min: 0, step: 0.0001 }}
                    fullWidth
                  />
                </Grid>
              )}

              {/* Leverage */}
              <Grid item xs={12} sm={6}>
                <TextField
                  label="Leverage Multiplier"
                  type="number"
                  value={tradeData.leverage}
                  onChange={(e) => setTradeData(prev => ({ ...prev, leverage: Number(e.target.value) }))}
                  inputProps={{ min: 1, max: 125, step: 1 }}
                  fullWidth
                />
              </Grid>

              {/* Margin Mode */}
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Margin Mode</InputLabel>
                  <Select
                    value={tradeData.margin_mode}
                    onChange={(e) => setTradeData(prev => ({ ...prev, margin_mode: e.target.value as 'cross' | 'isolated' }))}
                    label="Margin Mode"
                  >
                    <MenuItem value="cross">Cross Margin</MenuItem>
                    <MenuItem value="isolated">Isolated Margin</MenuItem>
                  </Select>
                </FormControl>
              </Grid>

              {/* Take Profit */}
              <Grid item xs={12} sm={6}>
                <TextField
                  label="Take Profit (%)"
                  type="number"
                  value={tradeData.take_profit_percentage}
                  onChange={(e) => setTradeData(prev => ({ ...prev, take_profit_percentage: Number(e.target.value) }))}
                  inputProps={{ min: 0.1, max: 50, step: 0.1 }}
                  fullWidth
                  helperText={calculateLeverageDisplay(tradeData.take_profit_percentage, tradeData.leverage, true)}
                />
              </Grid>

              {/* Stop Loss */}
              <Grid item xs={12} sm={6}>
                <TextField
                  label="Stop Loss (%)"
                  type="number"
                  value={tradeData.stop_loss_percentage}
                  onChange={(e) => setTradeData(prev => ({ ...prev, stop_loss_percentage: Number(e.target.value) }))}
                  inputProps={{ min: 0.1, max: 20, step: 0.1 }}
                  fullWidth
                  helperText={calculateLeverageDisplay(tradeData.stop_loss_percentage, tradeData.leverage, false)}
                />
              </Grid>
            </Grid>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTradeDialogOpen(false)}>Cancel</Button>
          <Button 
            variant="contained" 
            color={tradeData.side === 'buy' ? 'success' : 'error'}
            onClick={handleConfirmTrade}
            startIcon={<PlayArrow />}
          >
            Confirm {tradeData.side === 'buy' ? 'LONG' : 'SHORT'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Confirmation Dialog */}
      <Dialog open={confirmDialogOpen} onClose={() => setConfirmDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Warning color="warning" />
          Confirm Manual Trade
        </DialogTitle>
        <DialogContent>
          <Alert severity="warning" sx={{ mb: 2 }}>
            You are about to execute a manual trade. Please review all details carefully.
          </Alert>
          
          <Box sx={{ mt: 2 }}>
            <Typography variant="h6" gutterBottom>Trade Summary:</Typography>
            <Typography>• <strong>Coin:</strong> {selectedCoin}</Typography>
            <Typography>• <strong>Direction:</strong> {tradeData.side === 'buy' ? 'LONG 📈' : 'SHORT 📉'}</Typography>
            <Typography>• <strong>Order Type:</strong> {tradeData.order_type.toUpperCase()}</Typography>
            <Typography>• <strong>Amount:</strong> {calculateTradeAmounts().tradeAmount.toFixed(2)} USDT ({tradeData.amount_percentage}%)</Typography>
            <Typography>• <strong>Leverage:</strong> {tradeData.leverage}x ({tradeData.margin_mode})</Typography>
            <Typography>• <strong>Take Profit:</strong> {calculateLeverageDisplay(tradeData.take_profit_percentage, tradeData.leverage, true)}</Typography>
            <Typography>• <strong>Stop Loss:</strong> {calculateLeverageDisplay(tradeData.stop_loss_percentage, tradeData.leverage, false)}</Typography>
            {tradeData.order_type === 'limit' && (
              <Typography>• <strong>Limit Price:</strong> ${tradeData.limit_price?.toFixed(4)}</Typography>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmDialogOpen(false)}>Cancel</Button>
          <Button 
            variant="contained" 
            color={tradeData.side === 'buy' ? 'success' : 'error'}
            onClick={handleExecuteTrade}
            disabled={manualTradeMutation.isPending}
            startIcon={manualTradeMutation.isPending ? <CircularProgress size={20} /> : <PlayArrow />}
          >
            {manualTradeMutation.isPending ? 'Executing...' : `Execute ${tradeData.side.toUpperCase()}`}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};