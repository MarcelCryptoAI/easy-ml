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
  InputAdornment,
  Pagination,
  FormControl,
  InputLabel,
  Select,
  MenuItem
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
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(50);

  // Get all coins data
  const { data: coins = [], refetch: refetchCoins } = useQuery({
    queryKey: ['coins'],
    queryFn: tradingApi.getCoins,
    refetchInterval: 30000
  });

  // Get trading recommendations for all coins using batch endpoint
  const { data: dashboardData = [], isLoading, refetch } = useQuery({
    queryKey: ['compact-dashboard'],
    queryFn: async () => {
      try {
        // Use the new batch endpoint to get all dashboard data in one request
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/dashboard/batch`
        );
        
        if (!response.ok) {
          throw new Error('Failed to fetch dashboard data');
        }
        
        const data = await response.json();
        return data;
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
        // Return empty array on error
        return [];
      }
    },
    enabled: coins.length > 0,
    refetchInterval: 60000, // Refresh every minute
    staleTime: 30000 // Data is considered fresh for 30 seconds
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

  // Pagination calculations
  const totalPages = Math.ceil(filteredData.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const paginatedData = filteredData.slice(startIndex, endIndex);

  // Reset to first page when search changes
  useEffect(() => {
    setCurrentPage(1);
  }, [searchTerm, itemsPerPage]);

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

      {/* Search and Pagination Controls */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
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
          sx={{ width: 300 }}
        />
        
        <Box display="flex" alignItems="center" gap={2}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Per Page</InputLabel>
            <Select
              value={itemsPerPage}
              onChange={(e) => setItemsPerPage(e.target.value as number)}
              label="Per Page"
            >
              <MenuItem value={25}>25</MenuItem>
              <MenuItem value={50}>50</MenuItem>
              <MenuItem value={100}>100</MenuItem>
              <MenuItem value={filteredData.length}>All ({filteredData.length})</MenuItem>
            </Select>
          </FormControl>
          
          <Typography variant="body2" color="textSecondary">
            {startIndex + 1}-{Math.min(endIndex, filteredData.length)} of {filteredData.length}
          </Typography>
        </Box>
      </Box>

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
              {paginatedData.map((coin) => (
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
                    <Box display="flex" alignItems="center" gap={1}>
                      <Chip
                        label={`${coin.models_trained}/10`}
                        color={coin.models_trained === 10 ? 'success' : coin.models_trained >= 5 ? 'warning' : 'error'}
                        size="small"
                        variant="filled"
                      />
                      <LinearProgress 
                        variant="determinate" 
                        value={(coin.models_trained / 10) * 100}
                        sx={{ 
                          width: 40, 
                          height: 4,
                          borderRadius: 2
                        }}
                        color={coin.models_trained === 10 ? 'success' : 'primary'}
                      />
                    </Box>
                  </TableCell>
                  
                  <TableCell>
                    <Box display="flex" gap={0.5}>
                      <Chip label={`↗${coin.consensus_breakdown?.buy || 0}`} color="success" size="small" variant="outlined" />
                      <Chip label={`↘${coin.consensus_breakdown?.sell || 0}`} color="error" size="small" variant="outlined" />
                      <Chip label={`↔${coin.consensus_breakdown?.hold || 0}`} color="default" size="small" variant="outlined" />
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

      {/* Pagination */}
      {totalPages > 1 && (
        <Box display="flex" justifyContent="center" alignItems="center" mt={2} gap={2}>
          <Typography variant="body2" color="textSecondary">
            Page {currentPage} of {totalPages}
          </Typography>
          <Pagination
            count={totalPages}
            page={currentPage}
            onChange={(event, value) => setCurrentPage(value)}
            color="primary"
            size="medium"
            showFirstButton
            showLastButton
          />
        </Box>
      )}

      {/* No data message */}
      {!isLoading && filteredData.length === 0 && (
        <Alert severity="info" sx={{ mt: 2 }}>
          No trading data available. ML models are still training.
        </Alert>
      )}
    </Box>
  );
};