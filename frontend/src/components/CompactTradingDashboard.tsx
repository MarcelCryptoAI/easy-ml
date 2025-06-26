import React, { useState, useEffect } from 'react';
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
  TableSortLabel,
  Chip,
  IconButton,
  Button,
  Alert,
  LinearProgress,
  TextField,
  InputAdornment
} from '@mui/material';
import { Refresh, Search, TrendingUp, TrendingDown, TrendingFlat } from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { tradingApi } from '../utils/api';

interface CoinData {
  coin_symbol: string;
  recommendation: string;
  confidence: number;
  avg_confidence: number;
  models_trained: number;
  last_trained: string;
  consensus_breakdown: {
    buy: number;
    sell: number;
    hold: number;
  };
}

type SortField = 'coin_symbol' | 'recommendation' | 'confidence' | 'avg_confidence' | 'models_trained';
type SortDirection = 'asc' | 'desc';

export const CompactTradingDashboard: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [sortField, setSortField] = useState<SortField>('confidence');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  // Get all coins data
  const { data: coins = [], refetch: refetchCoins } = useQuery({
    queryKey: ['coins'],
    queryFn: tradingApi.getCoins,
    refetchInterval: 30000
  });

  // Get trading recommendations for all coins
  const { data: dashboardData = [], isLoading, refetch } = useQuery({
    queryKey: ['compact-dashboard'],
    queryFn: async () => {
      const coinDataPromises = coins.map(async (coin) => { // All coins, no limit
        try {
          const recommendation = await tradingApi.getRecommendation(coin.symbol);
          const predictions = await tradingApi.getPredictions(coin.symbol);
          
          return {
            coin_symbol: coin.symbol,
            recommendation: recommendation.recommendation,
            confidence: recommendation.confidence,
            avg_confidence: recommendation.avg_confidence,
            models_trained: predictions.length,
            last_trained: predictions[0]?.created_at || 'Never',
            consensus_breakdown: recommendation.consensus_breakdown
          };
        } catch (error) {
          return {
            coin_symbol: coin.symbol,
            recommendation: 'HOLD',
            confidence: 0,
            avg_confidence: 0,
            models_trained: 0,
            last_trained: 'Never',
            consensus_breakdown: { buy: 0, sell: 0, hold: 0 }
          };
        }
      });
      
      return Promise.all(coinDataPromises);
    },
    enabled: coins.length > 0,
    refetchInterval: 60000 // Refresh every minute
  });

  // Filter and sort data
  const filteredData = dashboardData
    .filter(coin => 
      coin.coin_symbol.toLowerCase().includes(searchTerm.toLowerCase())
    )
    .sort((a, b) => {
      const aValue = a[sortField];
      const bValue = b[sortField];
      
      if (sortDirection === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

  const handleSort = (field: SortField) => {
    if (field === sortField) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const getRecommendationIcon = (recommendation: string) => {
    switch (recommendation) {
      case 'LONG': return <TrendingUp color="success" />;
      case 'SHORT': return <TrendingDown color="error" />;
      default: return <TrendingFlat color="disabled" />;
    }
  };

  const getRecommendationColor = (recommendation: string) => {
    switch (recommendation) {
      case 'LONG': return 'success';
      case 'SHORT': return 'error';
      default: return 'default';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return 'success';
    if (confidence >= 60) return 'warning';
    return 'error';
  };

  return (
    <Box sx={{ p: 2 }}>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h5">🤖 Autonomous Trading Dashboard</Typography>
        <Box display="flex" gap={1}>
          <Button 
            variant="outlined" 
            onClick={() => tradingApi.optimizeAllStrategies()}
            size="small"
          >
            Optimize All
          </Button>
          <IconButton onClick={() => refetch()} disabled={isLoading}>
            <Refresh />
          </IconButton>
        </Box>
      </Box>

      {/* Search */}
      <TextField
        size="small"
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
        sx={{ mb: 2, width: 300 }}
      />

      {/* Stats Bar */}
      <Box display="flex" gap={2} mb={2}>
        <Chip label={`Total: ${filteredData.length}`} variant="outlined" />
        <Chip 
          label={`LONG: ${filteredData.filter(c => c.recommendation === 'LONG').length}`}
          color="success"
          size="small"
        />
        <Chip 
          label={`SHORT: ${filteredData.filter(c => c.recommendation === 'SHORT').length}`}
          color="error"
          size="small"
        />
        <Chip 
          label={`HOLD: ${filteredData.filter(c => c.recommendation === 'HOLD').length}`}
          color="default" 
          size="small"
        />
      </Box>

      {/* Loading */}
      {isLoading && <LinearProgress sx={{ mb: 2 }} />}

      {/* Compact Table */}
      <Paper sx={{ maxHeight: 600, overflow: 'auto' }}>
        <TableContainer>
          <Table stickyHeader size="small">
            <TableHead>
              <TableRow>
                <TableCell>
                  <TableSortLabel
                    active={sortField === 'coin_symbol'}
                    direction={sortField === 'coin_symbol' ? sortDirection : 'asc'}
                    onClick={() => handleSort('coin_symbol')}
                  >
                    Coin
                  </TableSortLabel>
                </TableCell>
                
                <TableCell>
                  <TableSortLabel
                    active={sortField === 'recommendation'}
                    direction={sortField === 'recommendation' ? sortDirection : 'asc'}
                    onClick={() => handleSort('recommendation')}
                  >
                    Signal
                  </TableSortLabel>
                </TableCell>
                
                <TableCell>
                  <TableSortLabel
                    active={sortField === 'confidence'}
                    direction={sortField === 'confidence' ? sortDirection : 'asc'}
                    onClick={() => handleSort('confidence')}
                  >
                    Confidence
                  </TableSortLabel>
                </TableCell>
                
                <TableCell>
                  <TableSortLabel
                    active={sortField === 'avg_confidence'}
                    direction={sortField === 'avg_confidence' ? sortDirection : 'asc'}
                    onClick={() => handleSort('avg_confidence')}
                  >
                    Avg ML
                  </TableSortLabel>
                </TableCell>
                
                <TableCell>
                  <TableSortLabel
                    active={sortField === 'models_trained'}
                    direction={sortField === 'models_trained' ? sortDirection : 'asc'}
                    onClick={() => handleSort('models_trained')}
                  >
                    Models
                  </TableSortLabel>
                </TableCell>
                
                <TableCell>Consensus</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredData.map((coin) => (
                <TableRow 
                  key={coin.coin_symbol}
                  hover
                  sx={{ 
                    backgroundColor: coin.confidence >= 75 
                      ? (coin.recommendation === 'LONG' ? '#e8f5e8' : coin.recommendation === 'SHORT' ? '#fde8e8' : 'inherit')
                      : 'inherit'
                  }}
                >
                  <TableCell>
                    <Typography variant="body2" fontWeight="bold">
                      {coin.coin_symbol}
                    </Typography>
                  </TableCell>
                  
                  <TableCell>
                    <Box display="flex" alignItems="center" gap={0.5}>
                      {getRecommendationIcon(coin.recommendation)}
                      <Chip
                        label={coin.recommendation}
                        color={getRecommendationColor(coin.recommendation) as any}
                        size="small"
                      />
                    </Box>
                  </TableCell>
                  
                  <TableCell>
                    <Chip
                      label={`${coin.confidence.toFixed(1)}%`}
                      color={getConfidenceColor(coin.confidence) as any}
                      size="small"
                    />
                  </TableCell>
                  
                  <TableCell>
                    <Typography variant="body2">
                      {coin.avg_confidence.toFixed(1)}%
                    </Typography>
                  </TableCell>
                  
                  <TableCell>
                    <Chip
                      label={`${coin.models_trained}/10`}
                      color={coin.models_trained === 10 ? 'success' : 'warning'}
                      size="small"
                      variant="outlined"
                    />
                  </TableCell>
                  
                  <TableCell>
                    <Box display="flex" gap={0.5}>
                      <Chip label={`↗${coin.consensus_breakdown.buy}`} color="success" size="small" variant="outlined" />
                      <Chip label={`↘${coin.consensus_breakdown.sell}`} color="error" size="small" variant="outlined" />
                      <Chip label={`↔${coin.consensus_breakdown.hold}`} color="default" size="small" variant="outlined" />
                    </Box>
                  </TableCell>
                  
                  <TableCell>
                    <Button 
                      size="small"
                      variant="outlined"
                      onClick={() => tradingApi.optimizeStrategy(coin.coin_symbol)}
                    >
                      Optimize
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      {/* No data message */}
      {!isLoading && filteredData.length === 0 && (
        <Alert severity="info" sx={{ mt: 2 }}>
          No trading data available. ML models are still training.
        </Alert>
      )}
    </Box>
  );
};