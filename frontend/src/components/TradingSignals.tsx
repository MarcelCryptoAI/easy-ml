import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  TextField,
  InputAdornment,
  Alert,
  LinearProgress,
  IconButton,
  Tooltip
} from '@mui/material';
import { Search, TrendingUp, TrendingDown, Remove, Refresh } from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';

interface TradingSignal {
  id: string;
  coin_symbol: string;
  signal_type: 'LONG' | 'SHORT' | 'HOLD';
  timestamp: string;
  models_agreed: number;
  total_models: number;
  avg_confidence: number;
  entry_price: number;
  current_price: number;
  position_size_usdt: number;
  status: 'open' | 'closed' | 'cancelled';
  unrealized_pnl_usdt: number;
  unrealized_pnl_percent: number;
  criteria_met: {
    confidence_threshold: boolean;
    model_agreement: boolean;
    risk_management: boolean;
  };
}

export const TradingSignals: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<'all' | 'open' | 'closed'>('all');

  const { data: signals = [], isLoading, refetch } = useQuery({
    queryKey: ['trading-signals'],
    queryFn: async () => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/signals`);
      if (!response.ok) {
        throw new Error(`Failed to fetch signals: ${response.status} ${response.statusText}`);
      }
      return await response.json() as TradingSignal[];
    },
    refetchInterval: 30000 // Refresh every 30 seconds
  });

  const filteredSignals = signals.filter(signal => {
    const matchesSearch = signal.coin_symbol.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === 'all' || signal.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const getSignalIcon = (type: string) => {
    switch (type) {
      case 'LONG': return <TrendingUp color="success" />;
      case 'SHORT': return <TrendingDown color="error" />;
      case 'HOLD': return <Remove color="disabled" />;
      default: return <Remove />;
    }
  };

  const getSignalColor = (type: string) => {
    switch (type) {
      case 'LONG': return 'success';
      case 'SHORT': return 'error';
      case 'HOLD': return 'default';
      default: return 'default';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'open': return 'warning';
      case 'closed': return 'success';
      case 'cancelled': return 'error';
      default: return 'default';
    }
  };

  const getPnLColor = (pnl: number) => {
    return pnl >= 0 ? 'success' : 'error';
  };

  const formatDateTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleString('nl-NL', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const totalPnL = filteredSignals.reduce((sum, signal) => sum + signal.unrealized_pnl_usdt, 0);
  const openPositions = filteredSignals.filter(s => s.status === 'open').length;

  return (
    <Box sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">📡 Trading Signals</Typography>
        <IconButton onClick={() => refetch()}>
          <Refresh />
        </IconButton>
      </Box>

      {/* Summary Stats */}
      <Box display="flex" gap={2} mb={3}>
        <Paper sx={{ p: 2, flex: 1 }}>
          <Typography variant="h6" color="primary">Total Signals</Typography>
          <Typography variant="h4">{signals.length}</Typography>
        </Paper>
        <Paper sx={{ p: 2, flex: 1 }}>
          <Typography variant="h6" color="warning.main">Open Positions</Typography>
          <Typography variant="h4">{openPositions}</Typography>
        </Paper>
        <Paper sx={{ p: 2, flex: 1 }}>
          <Typography variant="h6" color={totalPnL >= 0 ? 'success.main' : 'error.main'}>
            Total P&L (USDT)
          </Typography>
          <Typography variant="h4" color={totalPnL >= 0 ? 'success.main' : 'error.main'}>
            {totalPnL >= 0 ? '+' : ''}{totalPnL.toFixed(2)}
          </Typography>
        </Paper>
      </Box>

      <Alert severity="info" sx={{ mb: 3 }}>
        <strong>Signal Criteria:</strong> Only signals that meet ALL criteria are shown:
        <br />• Model Agreement: Required number of models agree on direction
        <br />• Confidence Threshold: Average confidence exceeds minimum threshold
        <br />• Risk Management: Position sizing and risk parameters validated
      </Alert>

      {/* Filters */}
      <Box display="flex" gap={2} mb={3}>
        <TextField
          placeholder="Search coins..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search />
              </InputAdornment>
            ),
          }}
          sx={{ minWidth: 300 }}
        />
        <TextField
          select
          label="Status"
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value as any)}
          SelectProps={{ native: true }}
          sx={{ minWidth: 150 }}
        >
          <option value="all">All Status</option>
          <option value="open">Open</option>
          <option value="closed">Closed</option>
        </TextField>
      </Box>

      {isLoading ? (
        <LinearProgress />
      ) : (
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell><strong>Date/Time</strong></TableCell>
                <TableCell><strong>Coin</strong></TableCell>
                <TableCell><strong>Signal</strong></TableCell>
                <TableCell><strong>Models</strong></TableCell>
                <TableCell><strong>Confidence</strong></TableCell>
                <TableCell><strong>Entry Price</strong></TableCell>
                <TableCell><strong>Current Price</strong></TableCell>
                <TableCell><strong>Position Size</strong></TableCell>
                <TableCell><strong>Status</strong></TableCell>
                <TableCell><strong>P&L (USDT)</strong></TableCell>
                <TableCell><strong>P&L (%)</strong></TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredSignals.map((signal) => (
                <TableRow key={signal.id}>
                  <TableCell>
                    <Typography variant="body2">
                      {formatDateTime(signal.timestamp)}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="subtitle2" fontWeight="bold">
                      {signal.coin_symbol}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip
                      icon={getSignalIcon(signal.signal_type)}
                      label={signal.signal_type}
                      color={getSignalColor(signal.signal_type) as any}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    <Tooltip title={`${signal.models_agreed} out of ${signal.total_models} models agreed`}>
                      <Chip
                        label={`${signal.models_agreed}/${signal.total_models}`}
                        color="info"
                        size="small"
                      />
                    </Tooltip>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {signal.avg_confidence.toFixed(1)}%
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      ${signal.entry_price.toFixed(4)}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      ${signal.current_price.toFixed(4)}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      ${signal.position_size_usdt.toFixed(0)}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={signal.status.toUpperCase()}
                      color={getStatusColor(signal.status) as any}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    <Typography 
                      variant="body2" 
                      color={getPnLColor(signal.unrealized_pnl_usdt)}
                      fontWeight="bold"
                    >
                      {signal.unrealized_pnl_usdt >= 0 ? '+' : ''}
                      {signal.unrealized_pnl_usdt.toFixed(2)}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography 
                      variant="body2" 
                      color={getPnLColor(signal.unrealized_pnl_percent)}
                      fontWeight="bold"
                    >
                      {signal.unrealized_pnl_percent >= 0 ? '+' : ''}
                      {signal.unrealized_pnl_percent.toFixed(2)}%
                    </Typography>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {filteredSignals.length === 0 && !isLoading && (
        <Alert severity="info" sx={{ mt: 2 }}>
          No trading signals found matching your criteria.
        </Alert>
      )}
    </Box>
  );
};