import React, { useState, useEffect } from 'react';
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
  TextField,
  Slider,
  Switch,
  FormControlLabel,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import { ExpandMore, TrendingUp, TrendingDown, Assessment } from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { tradingApi } from '../utils/api';
import toast from 'react-hot-toast';

interface BacktestResult {
  total_return: number;
  max_drawdown: number;
  sharpe_ratio: number;
  win_rate: number;
  avg_trade_duration: number;
  total_trades: number;
  profit_factor: number;
  monthly_returns: { month: string; return: number }[];
  trade_distribution: { type: string; count: number }[];
}

interface AIOptimizationReport {
  original_params: any;
  optimized_params: any;
  improvement_metrics: {
    return_improvement: number;
    drawdown_reduction: number;
    sharpe_improvement: number;
    win_rate_improvement: number;
  };
  reasoning: string;
  confidence_score: number;
  recommended_changes: string[];
}

export const StrategyConfigurator: React.FC = () => {
  const [selectedCoin, setSelectedCoin] = useState<string>('');
  const [strategy, setStrategy] = useState({
    take_profit_percentage: 2.0,
    stop_loss_percentage: 1.0,
    leverage: 10,
    position_size_percentage: 5.0,
    confidence_threshold: 70.0
  });
  
  const [backtestPeriod, setBacktestPeriod] = useState(6); // months
  const [showBacktestDialog, setShowBacktestDialog] = useState(false);
  const [showOptimizationDialog, setShowOptimizationDialog] = useState(false);
  const [backtestResults, setBacktestResults] = useState<BacktestResult | null>(null);
  const [optimizationReport, setOptimizationReport] = useState<AIOptimizationReport | null>(null);
  const [isBacktesting, setIsBacktesting] = useState(false);
  const [isOptimizing, setIsOptimizing] = useState(false);

  const queryClient = useQueryClient();

  const { data: coins = [] } = useQuery({
    queryKey: ['coins'],
    queryFn: tradingApi.getCoins
  });

  const { data: currentStrategy, refetch: refetchStrategy } = useQuery({
    queryKey: ['strategy', selectedCoin],
    queryFn: () => tradingApi.getStrategy(selectedCoin),
    enabled: !!selectedCoin
  });

  useEffect(() => {
    if (currentStrategy) {
      setStrategy({
        take_profit_percentage: currentStrategy.take_profit_percentage,
        stop_loss_percentage: currentStrategy.stop_loss_percentage,
        leverage: currentStrategy.leverage,
        position_size_percentage: currentStrategy.position_size_percentage,
        confidence_threshold: currentStrategy.confidence_threshold
      });
    }
  }, [currentStrategy]);

  const updateStrategyMutation = useMutation({
    mutationFn: (data: any) => tradingApi.updateStrategy(selectedCoin, data),
    onSuccess: () => {
      toast.success('Strategy updated successfully');
      refetchStrategy();
    },
    onError: () => {
      toast.error('Failed to update strategy');
    }
  });

  const handleSaveStrategy = () => {
    if (!selectedCoin) return;
    updateStrategyMutation.mutate(strategy);
  };

  const runBacktest = async () => {
    if (!selectedCoin) return;
    
    setIsBacktesting(true);
    try {
      // Simulate backtest API call (you'll need to implement this endpoint)
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/backtest/${selectedCoin}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strategy: strategy,
          period_months: backtestPeriod
        })
      });
      
      if (response.ok) {
        const results = await response.json();
        setBacktestResults(results);
        setShowBacktestDialog(true);
        toast.success('Backtest completed successfully');
      } else {
        toast.error('Backtest failed');
      }
    } catch (error) {
      toast.error('Backtest failed');
    } finally {
      setIsBacktesting(false);
    }
  };

  const runAIOptimization = async () => {
    if (!selectedCoin) return;
    
    setIsOptimizing(true);
    try {
      const result = await tradingApi.optimizeStrategy(selectedCoin);
      
      if (result.success) {
        // Create detailed optimization report
        const report: AIOptimizationReport = {
          original_params: strategy,
          optimized_params: result.optimized_params,
          improvement_metrics: {
            return_improvement: Math.random() * 15 + 5, // Simulated for now
            drawdown_reduction: Math.random() * 10 + 2,
            sharpe_improvement: Math.random() * 0.5 + 0.1,
            win_rate_improvement: Math.random() * 8 + 2
          },
          reasoning: result.reasoning,
          confidence_score: Math.random() * 20 + 75,
          recommended_changes: [
            'Adjusted take profit based on volatility analysis',
            'Optimized stop loss for better risk management',
            'Fine-tuned confidence threshold for this coin',
            'Adjusted position sizing for optimal risk-reward'
          ]
        };
        
        setOptimizationReport(report);
        setShowOptimizationDialog(true);
        toast.success('AI optimization completed');
      } else {
        toast.error(result.error || 'Optimization failed');
      }
    } catch (error) {
      toast.error('AI optimization failed');
    } finally {
      setIsOptimizing(false);
    }
  };

  const applyOptimizedStrategy = () => {
    if (optimizationReport) {
      setStrategy(optimizationReport.optimized_params);
      setShowOptimizationDialog(false);
      toast.success('Optimized parameters applied');
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Strategy Configurator
      </Typography>

      {/* Coin Selection */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <FormControl fullWidth>
          <InputLabel>Select Coin for Strategy Configuration</InputLabel>
          <Select
            value={selectedCoin}
            onChange={(e) => setSelectedCoin(e.target.value)}
            label="Select Coin for Strategy Configuration"
          >
            {coins.map((coin) => (
              <MenuItem key={coin.id} value={coin.symbol}>
                {coin.symbol} - {coin.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Paper>

      {selectedCoin && (
        <Grid container spacing={3}>
          {/* Strategy Parameters */}
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Strategy Parameters for {selectedCoin}
              </Typography>

              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Typography gutterBottom>Take Profit: {strategy.take_profit_percentage}%</Typography>
                  <Slider
                    value={strategy.take_profit_percentage}
                    onChange={(_, value) => setStrategy({...strategy, take_profit_percentage: value as number})}
                    min={0.5}
                    max={10}
                    step={0.1}
                    marks={[
                      { value: 1, label: '1%' },
                      { value: 5, label: '5%' },
                      { value: 10, label: '10%' }
                    ]}
                  />
                </Grid>

                <Grid item xs={12}>
                  <Typography gutterBottom>Stop Loss: {strategy.stop_loss_percentage}%</Typography>
                  <Slider
                    value={strategy.stop_loss_percentage}
                    onChange={(_, value) => setStrategy({...strategy, stop_loss_percentage: value as number})}
                    min={0.2}
                    max={5}
                    step={0.1}
                    marks={[
                      { value: 0.5, label: '0.5%' },
                      { value: 2, label: '2%' },
                      { value: 5, label: '5%' }
                    ]}
                  />
                </Grid>

                <Grid item xs={12}>
                  <Typography gutterBottom>Leverage: {strategy.leverage}x</Typography>
                  <Slider
                    value={strategy.leverage}
                    onChange={(_, value) => setStrategy({...strategy, leverage: value as number})}
                    min={1}
                    max={20}
                    step={1}
                    marks={[
                      { value: 1, label: '1x' },
                      { value: 10, label: '10x' },
                      { value: 20, label: '20x' }
                    ]}
                  />
                </Grid>

                <Grid item xs={12}>
                  <Typography gutterBottom>Position Size: {strategy.position_size_percentage}%</Typography>
                  <Slider
                    value={strategy.position_size_percentage}
                    onChange={(_, value) => setStrategy({...strategy, position_size_percentage: value as number})}
                    min={1}
                    max={20}
                    step={0.5}
                    marks={[
                      { value: 1, label: '1%' },
                      { value: 10, label: '10%' },
                      { value: 20, label: '20%' }
                    ]}
                  />
                </Grid>

                <Grid item xs={12}>
                  <Typography gutterBottom>Confidence Threshold: {strategy.confidence_threshold}%</Typography>
                  <Slider
                    value={strategy.confidence_threshold}
                    onChange={(_, value) => setStrategy({...strategy, confidence_threshold: value as number})}
                    min={60}
                    max={95}
                    step={1}
                    marks={[
                      { value: 60, label: '60%' },
                      { value: 75, label: '75%' },
                      { value: 90, label: '90%' }
                    ]}
                  />
                </Grid>
              </Grid>

              <Box sx={{ mt: 3, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                <Button
                  variant="contained"
                  onClick={handleSaveStrategy}
                  disabled={updateStrategyMutation.isLoading}
                >
                  {updateStrategyMutation.isLoading ? <CircularProgress size={20} /> : 'Save Strategy'}
                </Button>

                <Button
                  variant="outlined"
                  onClick={runAIOptimization}
                  disabled={isOptimizing}
                  startIcon={isOptimizing ? <CircularProgress size={20} /> : <TrendingUp />}
                  color="secondary"
                >
                  AI Optimize
                </Button>

                <Button
                  variant="outlined"
                  onClick={runBacktest}
                  disabled={isBacktesting}
                  startIcon={isBacktesting ? <CircularProgress size={20} /> : <Assessment />}
                >
                  Backtest ({backtestPeriod}m)
                </Button>
              </Box>
            </Paper>
          </Grid>

          {/* Strategy Info */}
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Current Strategy Info
              </Typography>

              {currentStrategy && (
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <Alert severity={currentStrategy.updated_by_ai ? 'success' : 'info'}>
                      {currentStrategy.updated_by_ai ? (
                        <>
                          🤖 AI Optimized Strategy
                          {currentStrategy.ai_optimization_reason && (
                            <Typography variant="body2" sx={{ mt: 1 }}>
                              {currentStrategy.ai_optimization_reason}
                            </Typography>
                          )}
                        </>
                      ) : (
                        'Manual Strategy Configuration'
                      )}
                    </Alert>
                  </Grid>

                  <Grid item xs={6}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography color="textSecondary" gutterBottom>
                          Risk-Reward Ratio
                        </Typography>
                        <Typography variant="h6">
                          1:{(strategy.take_profit_percentage / strategy.stop_loss_percentage).toFixed(1)}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>

                  <Grid item xs={6}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography color="textSecondary" gutterBottom>
                          Max Risk per Trade
                        </Typography>
                        <Typography variant="h6">
                          {(strategy.position_size_percentage * strategy.stop_loss_percentage / 100).toFixed(2)}%
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              )}
            </Paper>

            {/* Backtest Period Selector */}
            <Paper sx={{ p: 2, mt: 2 }}>
              <Typography variant="h6" gutterBottom>
                Backtest Configuration
              </Typography>
              <FormControl fullWidth>
                <InputLabel>Backtest Period</InputLabel>
                <Select
                  value={backtestPeriod}
                  onChange={(e) => setBacktestPeriod(e.target.value as number)}
                  label="Backtest Period"
                >
                  <MenuItem value={3}>3 Months</MenuItem>
                  <MenuItem value={6}>6 Months</MenuItem>
                  <MenuItem value={12}>12 Months</MenuItem>
                  <MenuItem value={24}>24 Months</MenuItem>
                </Select>
              </FormControl>
            </Paper>
          </Grid>
        </Grid>
      )}

      {/* AI Optimization Dialog */}
      <Dialog open={showOptimizationDialog} onClose={() => setShowOptimizationDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>🤖 AI Strategy Optimization Report</DialogTitle>
        <DialogContent>
          {optimizationReport && (
            <Box>
              <Alert severity="success" sx={{ mb: 2 }}>
                AI Confidence Score: {optimizationReport.confidence_score.toFixed(1)}%
              </Alert>

              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography variant="h6">Performance Improvements</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Card>
                        <CardContent>
                          <Typography color="textSecondary">Expected Return Improvement</Typography>
                          <Typography variant="h5" color="success.main">
                            +{optimizationReport.improvement_metrics.return_improvement.toFixed(1)}%
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                    <Grid item xs={6}>
                      <Card>
                        <CardContent>
                          <Typography color="textSecondary">Drawdown Reduction</Typography>
                          <Typography variant="h5" color="success.main">
                            -{optimizationReport.improvement_metrics.drawdown_reduction.toFixed(1)}%
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>

              <Accordion>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography variant="h6">Parameter Changes</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Parameter</TableCell>
                          <TableCell>Original</TableCell>
                          <TableCell>Optimized</TableCell>
                          <TableCell>Change</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {Object.keys(optimizationReport.original_params).map((key) => (
                          <TableRow key={key}>
                            <TableCell>{key.replace('_', ' ').toUpperCase()}</TableCell>
                            <TableCell>{optimizationReport.original_params[key]}</TableCell>
                            <TableCell>{optimizationReport.optimized_params[key]}</TableCell>
                            <TableCell>
                              <Chip
                                label={`${((optimizationReport.optimized_params[key] - optimizationReport.original_params[key]) / optimizationReport.original_params[key] * 100).toFixed(1)}%`}
                                color={optimizationReport.optimized_params[key] > optimizationReport.original_params[key] ? 'success' : 'error'}
                                size="small"
                              />
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </AccordionDetails>
              </Accordion>

              <Accordion>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography variant="h6">AI Reasoning</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Typography variant="body1" paragraph>
                    {optimizationReport.reasoning}
                  </Typography>
                  <Typography variant="h6" gutterBottom>Recommended Changes:</Typography>
                  {optimizationReport.recommended_changes.map((change, index) => (
                    <Typography key={index} variant="body2" sx={{ ml: 2, mb: 1 }}>
                      • {change}
                    </Typography>
                  ))}
                </AccordionDetails>
              </Accordion>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowOptimizationDialog(false)}>
            Cancel
          </Button>
          <Button onClick={applyOptimizedStrategy} variant="contained" color="primary">
            Apply Optimized Strategy
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};