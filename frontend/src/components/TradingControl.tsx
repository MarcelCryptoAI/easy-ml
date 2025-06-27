import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Switch,
  FormControlLabel,
  TextField,
  Alert,
  Grid,
  Divider,
  CircularProgress,
  Chip
} from '@mui/material';
import { 
  PlayArrow, 
  Stop, 
  Settings, 
  TrendingUp, 
  Warning,
  AccountBalance
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import toast from 'react-hot-toast';

interface TradingStatus {
  enabled: boolean;
  auto_start: boolean;
  balance: number;
  min_balance_required: number;
  daily_loss_tracker: number;
  max_daily_loss: number;
  trades_today: number;
  last_reset_date: string;
}

interface RiskSettings {
  min_balance_required: number;
  max_daily_loss_percentage: number;
  auto_start_trading: boolean;
}

export const TradingControl: React.FC = () => {
  const [riskDialogOpen, setRiskDialogOpen] = useState(false);
  const [riskSettings, setRiskSettings] = useState<RiskSettings>({
    min_balance_required: 10,
    max_daily_loss_percentage: 5,
    auto_start_trading: true
  });
  const queryClient = useQueryClient();

  const { data: tradingStatus, isLoading } = useQuery({
    queryKey: ['trading-status'],
    queryFn: async () => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/trading/status`);
      if (!response.ok) throw new Error('Failed to fetch trading status');
      return await response.json() as TradingStatus;
    },
    refetchInterval: 5000 // Refresh every 5 seconds
  });

  const toggleTradingMutation = useMutation({
    mutationFn: async (enable: boolean) => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/trading/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enable })
      });
      if (!response.ok) throw new Error('Failed to toggle trading');
      return response.json();
    },
    onSuccess: (data, enable) => {
      toast.success(`Trading ${enable ? 'enabled' : 'disabled'}`);
      queryClient.invalidateQueries({ queryKey: ['trading-status'] });
    },
    onError: () => {
      toast.error('Failed to toggle trading');
    }
  });

  const forceStartMutation = useMutation({
    mutationFn: async () => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/trading/force-start`, {
        method: 'POST'
      });
      if (!response.ok) throw new Error('Failed to force start trading');
      return response.json();
    },
    onSuccess: () => {
      toast.success('🚀 Trading force-started!');
      queryClient.invalidateQueries({ queryKey: ['trading-status'] });
    },
    onError: () => {
      toast.error('Failed to force start trading');
    }
  });

  const updateRiskSettingsMutation = useMutation({
    mutationFn: async (settings: RiskSettings) => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/trading/risk-settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
      });
      if (!response.ok) throw new Error('Failed to update risk settings');
      return response.json();
    },
    onSuccess: () => {
      toast.success('Risk settings updated');
      queryClient.invalidateQueries({ queryKey: ['trading-status'] });
    },
    onError: () => {
      toast.error('Failed to update risk settings');
    }
  });

  const handleToggleTrading = () => {
    if (tradingStatus) {
      toggleTradingMutation.mutate(!tradingStatus.enabled);
    }
  };

  const handleForceStart = () => {
    forceStartMutation.mutate();
  };

  const handleUpdateRiskSettings = () => {
    updateRiskSettingsMutation.mutate(riskSettings);
  };

  if (isLoading) {
    return (
      <Box sx={{ p: 3, display: 'flex', justifyContent: 'center' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (!tradingStatus) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error">Failed to load trading status</Alert>
      </Box>
    );
  }

  const isBalanceLow = tradingStatus.balance < tradingStatus.min_balance_required;
  const isDailyLossHigh = tradingStatus.daily_loss_tracker >= tradingStatus.max_daily_loss;
  const canTrade = !isBalanceLow && !isDailyLossHigh;

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" mb={3}>🎮 Trading Control</Typography>

      <Grid container spacing={3}>
        {/* Trading Status */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" mb={2}>Trading Status</Typography>
            
            <Box display="flex" alignItems="center" gap={2} mb={2}>
              <Chip 
                icon={tradingStatus.enabled ? <PlayArrow /> : <Stop />}
                label={tradingStatus.enabled ? 'ENABLED' : 'DISABLED'}
                color={tradingStatus.enabled ? 'success' : 'error'}
              />
              <Chip 
                icon={<AccountBalance />}
                label={`${tradingStatus.balance.toFixed(2)} USDT`}
                color={isBalanceLow ? 'error' : 'primary'}
              />
            </Box>

            <Box mb={3}>
              <FormControlLabel
                control={
                  <Switch 
                    checked={tradingStatus.enabled}
                    onChange={handleToggleTrading}
                    disabled={toggleTradingMutation.isPending}
                  />
                }
                label="Enable Trading"
              />
            </Box>

            <Button
              variant="contained"
              color="warning"
              onClick={handleForceStart}
              disabled={forceStartMutation.isPending || tradingStatus.enabled}
              startIcon={<TrendingUp />}
              fullWidth
              sx={{ mb: 2 }}
            >
              Force Start Trading
            </Button>

            {!canTrade && (
              <Alert severity="warning" sx={{ mb: 2 }}>
                {isBalanceLow && `Balance too low (${tradingStatus.balance.toFixed(2)} < ${tradingStatus.min_balance_required} USDT)`}
                {isDailyLossHigh && `Daily loss limit exceeded (${tradingStatus.daily_loss_tracker.toFixed(2)}%)`}
              </Alert>
            )}
          </Paper>
        </Grid>

        {/* Daily Statistics */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" mb={2}>Daily Statistics</Typography>
            
            <Box mb={2}>
              <Typography variant="body2" color="textSecondary">Trades Today</Typography>
              <Typography variant="h5">{tradingStatus.trades_today}</Typography>
            </Box>

            <Box mb={2}>
              <Typography variant="body2" color="textSecondary">Daily P&L</Typography>
              <Typography 
                variant="h5" 
                color={tradingStatus.daily_loss_tracker >= 0 ? 'success.main' : 'error.main'}
              >
                {tradingStatus.daily_loss_tracker >= 0 ? '+' : ''}{tradingStatus.daily_loss_tracker.toFixed(2)}%
              </Typography>
            </Box>

            <Box>
              <Typography variant="body2" color="textSecondary">
                Last Reset: {new Date(tradingStatus.last_reset_date).toLocaleDateString()}
              </Typography>
            </Box>
          </Paper>
        </Grid>

        {/* Risk Management */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
              <Typography variant="h6">Risk Management</Typography>
              <Button
                variant="outlined"
                startIcon={<Settings />}
                onClick={() => setRiskDialogOpen(!riskDialogOpen)}
              >
                Settings
              </Button>
            </Box>

            {riskDialogOpen && (
              <Box sx={{ border: '1px solid #ddd', borderRadius: 1, p: 2, mb: 2 }}>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={4}>
                    <TextField
                      label="Min Balance Required (USDT)"
                      type="number"
                      value={riskSettings.min_balance_required}
                      onChange={(e) => setRiskSettings({
                        ...riskSettings,
                        min_balance_required: Number(e.target.value)
                      })}
                      fullWidth
                    />
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <TextField
                      label="Max Daily Loss (%)"
                      type="number"
                      value={riskSettings.max_daily_loss_percentage}
                      onChange={(e) => setRiskSettings({
                        ...riskSettings,
                        max_daily_loss_percentage: Number(e.target.value)
                      })}
                      fullWidth
                    />
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={riskSettings.auto_start_trading}
                          onChange={(e) => setRiskSettings({
                            ...riskSettings,
                            auto_start_trading: e.target.checked
                          })}
                        />
                      }
                      label="Auto-start Trading"
                    />
                  </Grid>
                </Grid>
                <Box mt={2}>
                  <Button
                    variant="contained"
                    onClick={handleUpdateRiskSettings}
                    disabled={updateRiskSettingsMutation.isPending}
                  >
                    Update Settings
                  </Button>
                </Box>
              </Box>
            )}

            <Grid container spacing={2}>
              <Grid item xs={12} sm={4}>
                <Box>
                  <Typography variant="body2" color="textSecondary">Min Balance</Typography>
                  <Typography variant="h6">{tradingStatus.min_balance_required} USDT</Typography>
                </Box>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Box>
                  <Typography variant="body2" color="textSecondary">Max Daily Loss</Typography>
                  <Typography variant="h6">{tradingStatus.max_daily_loss}%</Typography>
                </Box>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Box>
                  <Typography variant="body2" color="textSecondary">Auto-start</Typography>
                  <Chip 
                    label={tradingStatus.auto_start ? 'Enabled' : 'Disabled'}
                    color={tradingStatus.auto_start ? 'success' : 'default'}
                    size="small"
                  />
                </Box>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};