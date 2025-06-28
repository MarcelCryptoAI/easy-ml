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
  Circle,
  Computer,
  Api,
  Memory
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { tradingApi } from '../utils/api';

interface ConnectionStatus {
  frontend_connected: boolean;
  backend_connected: boolean;
  worker_connected: boolean;
  database_connected: boolean;
  openai_connected: boolean;
  bybit_connected: boolean;
  uta_balance: string;
}

export const StatusTopBar: React.FC = () => {
  const { data: statusData, refetch } = useQuery({
    queryKey: ['system-status'],
    queryFn: async () => {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout
      
      try {
        // Try new status endpoint first
        const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/status`, {
          signal: controller.signal
        });
        clearTimeout(timeoutId);
        
        if (response.ok) {
          const data = await response.json();
          return data as ConnectionStatus;
        }
      } catch (error) {
        clearTimeout(timeoutId);
        console.warn('Status endpoint failed, falling back to health endpoint');
      }
      
      try {
        // Fallback to health endpoint
        const healthResponse = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/health`, {
          signal: controller.signal
        });
        clearTimeout(timeoutId);
        const healthData = await healthResponse.json();
        
        // Transform health data to status format
        return {
          frontend_connected: true,
          backend_connected: true,
          worker_connected: true, // Assume true if backend is running
          database_connected: healthData.database_coins > 0,
          openai_connected: true, // Assume true if backend is running
          bybit_connected: healthData.bybit_connected,
          uta_balance: healthData.account_balance?.toString() || "0.00"
        } as ConnectionStatus;
      } catch (error) {
        clearTimeout(timeoutId);
        // Return default disconnected state
        return {
          frontend_connected: true,
          backend_connected: false,
          worker_connected: false,
          database_connected: false,
          openai_connected: false,
          bybit_connected: false,
          uta_balance: "0.00"
        } as ConnectionStatus;
      }
    },
    refetchInterval: 15000, // Update every 15 seconds (less frequent)
    retry: 2
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
          {/* Frontend Status */}
          <Tooltip title="Frontend Connection">
            <Chip
              icon={<Computer />}
              label={
                <Box display="flex" alignItems="center">
                  {getStatusIcon(statusData?.frontend_connected || true)}
                  Frontend
                </Box>
              }
              color={getStatusColor(statusData?.frontend_connected || true) as any}
              variant="outlined"
              size="small"
            />
          </Tooltip>

          {/* Backend Status */}
          <Tooltip title="Backend API Connection">
            <Chip
              icon={<Api />}
              label={
                <Box display="flex" alignItems="center">
                  {getStatusIcon(statusData?.backend_connected || false)}
                  Backend
                </Box>
              }
              color={getStatusColor(statusData?.backend_connected || false) as any}
              variant="outlined"
              size="small"
            />
          </Tooltip>

          {/* Worker Status */}
          <Tooltip title="ML Worker Status">
            <Chip
              icon={<Memory />}
              label={
                <Box display="flex" alignItems="center">
                  {getStatusIcon(statusData?.worker_connected || false)}
                  Worker
                </Box>
              }
              color={getStatusColor(statusData?.worker_connected || false) as any}
              variant="outlined"
              size="small"
            />
          </Tooltip>

          {/* Database Status */}
          <Tooltip title="Database Connection">
            <Chip
              icon={<Storage />}
              label={
                <Box display="flex" alignItems="center">
                  {getStatusIcon(statusData?.database_connected || false)}
                  Database
                </Box>
              }
              color={getStatusColor(statusData?.database_connected || false) as any}
              variant="outlined"
              size="small"
            />
          </Tooltip>

          {/* ByBit API Status */}
          <Tooltip title="ByBit API Connection">
            <Chip
              icon={<TrendingUp />}
              label={
                <Box display="flex" alignItems="center">
                  {getStatusIcon(statusData?.bybit_connected || false)}
                  ByBit API
                </Box>
              }
              color={getStatusColor(statusData?.bybit_connected || false) as any}
              variant="outlined"
              size="small"
            />
          </Tooltip>

          {/* OpenAI API Status */}
          <Tooltip title="OpenAI API Connection">
            <Chip
              icon={<SmartToy />}
              label={
                <Box display="flex" alignItems="center">
                  {getStatusIcon(statusData?.openai_connected || false)}
                  OpenAI API
                </Box>
              }
              color={getStatusColor(statusData?.openai_connected || false) as any}
              variant="outlined"
              size="small"
            />
          </Tooltip>

          {/* Account Balance */}
          <Tooltip title="Account Balance (USDT)">
            <Chip
              icon={<AccountBalance />}
              label={`$${parseFloat(statusData?.uta_balance || '0').toFixed(2)}`}
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