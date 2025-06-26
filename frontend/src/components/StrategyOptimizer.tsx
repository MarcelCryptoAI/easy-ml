import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Switch,
  FormControlLabel,
  TextField,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import { 
  AutoAwesome, 
  Settings, 
  PlayArrow, 
  Stop, 
  Schedule,
  ExpandMore,
  TrendingUp,
  TrendingDown
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { tradingApi } from '../utils/api';
import toast from 'react-hot-toast';

interface OptimizationJob {
  coin_symbol: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  progress: number;
  current_strategy: any;
  optimized_strategy: any;
  improvement: {
    return_improvement: number;
    drawdown_reduction: number;
    sharpe_improvement: number;
  };
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
      // Mock data - replace with actual API
      const mockSession: OptimizationSession = {
        is_running: false,
        total_coins: 500,
        completed_coins: 0,
        current_coin: '',
        session_start_time: '',
        estimated_completion_time: '',
        auto_apply_optimizations: true
      };
      return mockSession;
    },
    refetchInterval: 5000
  });

  const { data: optimizationQueue = [] } = useQuery({
    queryKey: ['optimization-queue'],
    queryFn: async () => {
      // Mock data - replace with actual API
      const mockQueue: OptimizationJob[] = [];
      return mockQueue;
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
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        🎯 Strategy Optimizer
      </Typography>

      {/* Main Control Panel */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Bulk Strategy Optimization
              </Typography>
              
              {optimizationSession?.is_running ? (
                <Box>
                  <Typography variant="body2" color="textSecondary" gutterBottom>
                    Optimizing: {optimizationSession.current_coin}
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={(optimizationSession.completed_coins / optimizationSession.total_coins) * 100}
                    sx={{ mb: 2 }}
                  />
                  <Typography variant="caption" color="textSecondary">
                    {optimizationSession.completed_coins} / {optimizationSession.total_coins} coins completed
                  </Typography>
                  <Box mt={2}>
                    <Button
                      variant="contained"
                      color="error"
                      startIcon={<Stop />}
                      onClick={() => stopOptimizationMutation.mutate()}
                      disabled={stopOptimizationMutation.isPending}
                    >
                      Stop Optimization
                    </Button>
                  </Box>
                </Box>
              ) : (
                <Box>
                  <Typography variant="body2" color="textSecondary" gutterBottom>
                    Optimize strategies for all {coins.length} coins using AI and backtesting
                  </Typography>
                  <Box display="flex" gap={1} flexWrap="wrap" mt={2}>
                    <Button
                      variant="contained"
                      color="primary"
                      size="large"
                      startIcon={startOptimizeAllMutation.isPending ? <CircularProgress size={20} /> : <AutoAwesome />}
                      onClick={() => startOptimizeAllMutation.mutate()}
                      disabled={startOptimizeAllMutation.isPending}
                    >
                      Optimize All Strategies
                    </Button>
                    <Button
                      variant="outlined"
                      startIcon={<Settings />}
                      onClick={() => setShowSettingsDialog(true)}
                    >
                      Settings
                    </Button>
                  </Box>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Autonomous Optimization
              </Typography>
              <Typography variant="body2" color="textSecondary" gutterBottom>
                Automatically optimize strategies every {autoOptimizeSettings.interval_hours} hours
              </Typography>
              
              <FormControlLabel
                control={
                  <Switch
                    checked={autoOptimizeSettings.enabled}
                    onChange={(e) => setAutoOptimizeSettings({
                      ...autoOptimizeSettings,
                      enabled: e.target.checked
                    })}
                  />
                }
                label="Enable Auto-Optimization"
              />
              
              <Box mt={2}>
                <Button
                  variant="outlined"
                  color="secondary"
                  startIcon={<Schedule />}
                  onClick={() => setShowSettingsDialog(true)}
                >
                  Configure Schedule
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Default Strategies Setup */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Default Strategy Setup
          </Typography>
          <Typography variant="body2" color="textSecondary" gutterBottom>
            Create default trading strategies for all coins that don't have one yet
          </Typography>
          <Button
            variant="outlined"
            onClick={() => createDefaultStrategiesMutation.mutate()}
            disabled={createDefaultStrategiesMutation.isPending}
            startIcon={createDefaultStrategiesMutation.isPending ? <CircularProgress size={20} /> : <PlayArrow />}
          >
            Create Default Strategies
          </Button>
        </CardContent>
      </Card>

      {/* Optimization Queue */}
      {optimizationQueue.length > 0 && (
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography variant="h6">
              Optimization Queue ({optimizationQueue.length} coins)
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Position</TableCell>
                    <TableCell>Coin</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Progress</TableCell>
                    <TableCell>Expected Improvement</TableCell>
                    <TableCell>Started</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {optimizationQueue.map((job, index) => (
                    <TableRow key={job.coin_symbol}>
                      <TableCell>#{job.queue_position}</TableCell>
                      <TableCell>
                        <Typography variant="subtitle2" fontWeight="bold">
                          {job.coin_symbol}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip 
                          label={job.status}
                          color={getStatusColor(job.status)}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Box width={100}>
                          <LinearProgress 
                            variant="determinate" 
                            value={job.progress}
                            sx={{ mb: 0.5 }}
                          />
                          <Typography variant="caption">
                            {job.progress}%
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        {job.improvement && (
                          <Box display="flex" gap={0.5}>
                            <Chip
                              label={`+${job.improvement.return_improvement.toFixed(1)}%`}
                              color="success"
                              size="small"
                              icon={<TrendingUp />}
                            />
                          </Box>
                        )}
                      </TableCell>
                      <TableCell>
                        <Typography variant="caption">
                          {job.started_at ? new Date(job.started_at).toLocaleTimeString() : '-'}
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </AccordionDetails>
        </Accordion>
      )}

      {/* Settings Dialog */}
      <Dialog open={showSettingsDialog} onClose={() => setShowSettingsDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Auto-Optimization Settings</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={autoOptimizeSettings.enabled}
                    onChange={(e) => setAutoOptimizeSettings({
                      ...autoOptimizeSettings,
                      enabled: e.target.checked
                    })}
                  />
                }
                label="Enable Autonomous Optimization"
              />
            </Grid>
            
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Optimization Interval (hours)"
                type="number"
                value={autoOptimizeSettings.interval_hours}
                onChange={(e) => setAutoOptimizeSettings({
                  ...autoOptimizeSettings,
                  interval_hours: parseInt(e.target.value)
                })}
                helperText="How often to automatically optimize all strategies"
              />
            </Grid>
            
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Minimum Improvement Threshold (%)"
                type="number"
                value={autoOptimizeSettings.min_improvement_threshold}
                onChange={(e) => setAutoOptimizeSettings({
                  ...autoOptimizeSettings,
                  min_improvement_threshold: parseFloat(e.target.value)
                })}
                helperText="Only apply optimizations with at least this much improvement"
              />
            </Grid>
            
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={autoOptimizeSettings.auto_apply}
                    onChange={(e) => setAutoOptimizeSettings({
                      ...autoOptimizeSettings,
                      auto_apply: e.target.checked
                    })}
                  />
                }
                label="Automatically Apply Optimizations"
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowSettingsDialog(false)}>
            Cancel
          </Button>
          <Button 
            onClick={() => enableAutoOptimizeMutation.mutate(autoOptimizeSettings)}
            variant="contained"
            disabled={enableAutoOptimizeMutation.isPending}
          >
            Save Settings
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};