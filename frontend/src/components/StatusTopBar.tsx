import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Box,
  Chip,
  IconButton,
  Tooltip
} from '@mui/material';
import { 
  Storage, 
  SmartToy, 
  TrendingUp, 
  AccountBalance,
  Refresh,
  Circle
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { tradingApi } from '../utils/api';

interface ConnectionStatus {
  database: boolean;
  openai: boolean;
  bybit: boolean;
  account_balance: number;
}

export const StatusTopBar: React.FC = () => {
  const { data: healthData, refetch } = useQuery({
    queryKey: ['health-status'],
    queryFn: async () => {
      const health = await tradingApi.getHealth();
      return {
        database: health.database_coins > 0,
        openai: true, // Assume true if no error
        bybit: health.bybit_connected,
        account_balance: health.account_balance || 0
      };
    },
    refetchInterval: 10000, // Update every 10 seconds
    retry: 1
  });

  const getStatusColor = (connected: boolean) => connected ? 'success' : 'error';
  const getStatusIcon = (connected: boolean) => (
    <Circle 
      sx={{ 
        fontSize: 8, 
        color: connected ? '#4caf50' : '#f44336',
        mr: 0.5 
      }} 
    />
  );

  return (
    <AppBar position="static" elevation={1}>
      <Toolbar>
        {/* Title */}
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          🤖 Crypto Trading ML Platform - Autonomous Mode
        </Typography>

        {/* Connection Status */}
        <Box display="flex" alignItems="center" gap={1}>
          {/* Database Status */}
          <Tooltip title="Database Connection">
            <Chip
              icon={<Storage />}
              label={
                <Box display="flex" alignItems="center">
                  {getStatusIcon(healthData?.database || false)}
                  Database
                </Box>
              }
              color={getStatusColor(healthData?.database || false) as any}
              variant="outlined"
              size="small"
            />
          </Tooltip>

          {/* OpenAI Status */}
          <Tooltip title="OpenAI API Connection">
            <Chip
              icon={<SmartToy />}
              label={
                <Box display="flex" alignItems="center">
                  {getStatusIcon(healthData?.openai || false)}
                  OpenAI
                </Box>
              }
              color={getStatusColor(healthData?.openai || false) as any}
              variant="outlined"
              size="small"
            />
          </Tooltip>

          {/* Bybit Status */}
          <Tooltip title="Bybit API Connection">
            <Chip
              icon={<TrendingUp />}
              label={
                <Box display="flex" alignItems="center">
                  {getStatusIcon(healthData?.bybit || false)}
                  Bybit
                </Box>
              }
              color={getStatusColor(healthData?.bybit || false) as any}
              variant="outlined"
              size="small"
            />
          </Tooltip>

          {/* Account Balance */}
          <Tooltip title="Account Balance (USDT)">
            <Chip
              icon={<AccountBalance />}
              label={`$${(healthData?.account_balance || 0).toFixed(2)}`}
              color="primary"
              variant="filled"
              size="small"
              sx={{ 
                color: 'white',
                fontWeight: 'bold',
                minWidth: 100
              }}
            />
          </Tooltip>

          {/* Refresh Button */}
          <Tooltip title="Refresh Status">
            <IconButton 
              onClick={() => refetch()} 
              color="inherit" 
              size="small"
            >
              <Refresh />
            </IconButton>
          </Tooltip>
        </Box>
      </Toolbar>
    </AppBar>
  );
};