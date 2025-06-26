import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  Button,
  Switch,
  FormControlLabel,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Alert
} from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { tradingApi } from '../utils/api';
import { useWebSocket } from '../hooks/useWebSocket';
import { Trade, Position, Coin } from '../types';
import toast from 'react-hot-toast';

export const Dashboard: React.FC = () => {
  const [tradingEnabled, setTradingEnabled] = useState(false);
  const queryClient = useQueryClient();
  const { isConnected, subscribe, unsubscribe } = useWebSocket();

  const { data: coins = [] } = useQuery({
    queryKey: ['coins'],
    queryFn: tradingApi.getCoins,
    refetchInterval: 30000
  });

  const { data: openTrades = [] } = useQuery({
    queryKey: ['trades', 'open'],
    queryFn: () => tradingApi.getTrades('open'),
    refetchInterval: 5000
  });

  const { data: closedTrades = [] } = useQuery({
    queryKey: ['trades', 'closed'],
    queryFn: () => tradingApi.getTrades('closed'),
    refetchInterval: 10000
  });

  const { data: positionsData } = useQuery({
    queryKey: ['positions'],
    queryFn: tradingApi.getPositions,
    refetchInterval: 5000
  });

  const positions = positionsData?.positions || [];

  useEffect(() => {
    subscribe('trade_update', (data) => {
      queryClient.invalidateQueries({ queryKey: ['trades'] });
      queryClient.invalidateQueries({ queryKey: ['positions'] });
      
      if (data.action === 'opened') {
        toast.success(`Trade opened: ${data.symbol} ${data.side} at ${data.price}`);
      } else if (data.action === 'closed') {
        toast.success(`Trade closed: ${data.symbol} PnL: ${data.pnl?.toFixed(2)}%`);
      }
    });

    subscribe('prediction_update', () => {
      toast.info('New ML predictions available');
    });

    return () => {
      unsubscribe('trade_update');
      unsubscribe('prediction_update');
    };
  }, [subscribe, unsubscribe, queryClient]);

  const handleToggleTrading = async () => {
    try {
      const result = await tradingApi.toggleTrading(!tradingEnabled);
      if (result.success) {
        setTradingEnabled(!tradingEnabled);
        toast.success(result.message);
      }
    } catch (error) {
      toast.error('Failed to toggle trading');
    }
  };

  const handleBatchOptimize = async () => {
    try {
      toast.loading('Running AI optimization...');
      const result = await tradingApi.batchOptimize();
      if (result.success) {
        toast.success('Strategies optimized successfully');
      } else {
        toast.error('Optimization failed');
      }
    } catch (error) {
      toast.error('Failed to run optimization');
    }
  };

  const totalPnL = closedTrades.reduce((sum: number, trade: Trade) => sum + trade.pnl, 0);
  const winRate = closedTrades.length > 0 
    ? (closedTrades.filter((trade: Trade) => trade.pnl > 0).length / closedTrades.length) * 100 
    : 0;

  const pnlChartData = closedTrades.slice(-20).map((trade: Trade, index: number) => ({
    index: index + 1,
    pnl: trade.pnl,
    cumulative: closedTrades.slice(0, closedTrades.indexOf(trade) + 1)
      .reduce((sum: number, t: Trade) => sum + t.pnl, 0)
  }));

  return (
    <Box sx={{ p: 3 }}>
      <Grid container spacing={3}>
        {/* Header Controls */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Box display="flex" justifyContent="space-between" alignItems="center">
              <Typography variant="h4">Crypto Trading Dashboard</Typography>
              <Box display="flex" gap={2} alignItems="center">
                <Chip 
                  label={isConnected ? 'Connected' : 'Disconnected'} 
                  color={isConnected ? 'success' : 'error'} 
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={tradingEnabled}
                      onChange={handleToggleTrading}
                      color="primary"
                    />
                  }
                  label="Auto Trading"
                />
                <Button
                  variant="contained"
                  onClick={handleBatchOptimize}
                  color="secondary"
                >
                  AI Optimize All
                </Button>
              </Box>
            </Box>
          </Paper>
        </Grid>

        {/* Statistics Cards */}
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total P&L
              </Typography>
              <Typography variant="h5" color={totalPnL >= 0 ? 'success.main' : 'error.main'}>
                {totalPnL.toFixed(2)}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Win Rate
              </Typography>
              <Typography variant="h5">
                {winRate.toFixed(1)}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Active Positions
              </Typography>
              <Typography variant="h5">
                {positions.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Trades
              </Typography>
              <Typography variant="h5">
                {closedTrades.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* P&L Chart */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              P&L Performance
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={pnlChartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="index" />
                <YAxis />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="cumulative" 
                  stroke="#8884d8" 
                  strokeWidth={2}
                  name="Cumulative P&L"
                />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Active Positions */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Active Positions
            </Typography>
            {positions.length === 0 ? (
              <Alert severity="info">No active positions</Alert>
            ) : (
              <Box>
                {positions.map((position: Position) => (
                  <Card key={position.symbol} sx={{ mb: 1 }}>
                    <CardContent sx={{ py: 1 }}>
                      <Typography variant="subtitle2">{position.symbol}</Typography>
                      <Typography variant="body2" color="textSecondary">
                        {position.side} | Size: {position.size}
                      </Typography>
                      <Typography 
                        variant="body2" 
                        color={position.unrealisedPnl >= 0 ? 'success.main' : 'error.main'}
                      >
                        P&L: {position.unrealisedPnl.toFixed(2)}
                      </Typography>
                    </CardContent>
                  </Card>
                ))}
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Recent Trades */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Recent Trades
            </Typography>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Symbol</TableCell>
                    <TableCell>Side</TableCell>
                    <TableCell>Size</TableCell>
                    <TableCell>Price</TableCell>
                    <TableCell>P&L</TableCell>
                    <TableCell>Confidence</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Date</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {[...openTrades, ...closedTrades.slice(-10)].map((trade: Trade) => (
                    <TableRow key={trade.id}>
                      <TableCell>{trade.coin_symbol}</TableCell>
                      <TableCell>
                        <Chip 
                          label={trade.side} 
                          color={trade.side === 'buy' ? 'success' : 'error'} 
                          size="small"
                        />
                      </TableCell>
                      <TableCell>{trade.size.toFixed(4)}</TableCell>
                      <TableCell>${trade.price.toFixed(2)}</TableCell>
                      <TableCell 
                        sx={{ color: trade.pnl >= 0 ? 'success.main' : 'error.main' }}
                      >
                        {trade.pnl.toFixed(2)}%
                      </TableCell>
                      <TableCell>{trade.ml_confidence.toFixed(1)}%</TableCell>
                      <TableCell>
                        <Chip 
                          label={trade.status} 
                          color={trade.status === 'open' ? 'primary' : 'default'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        {new Date(trade.opened_at).toLocaleDateString()}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};