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
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  IconButton,
  Alert,
  Pagination,
  InputAdornment,
  LinearProgress
} from '@mui/material';
import { Edit, Settings, Search } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import toast from 'react-hot-toast';

interface CoinStrategy {
  coin_symbol: string;
  leverage: number;
  margin_mode: 'cross' | 'isolated';
  position_size_percent: number;
  confidence_threshold: number;
  min_models_required: number;
  total_models_available: number;
  take_profit_percentage: number;
  stop_loss_percentage: number;
  is_active: boolean;
  last_updated?: string;
  ai_optimized: boolean;
}

interface EditStrategyData {
  coin_symbol: string;
  leverage: number;
  margin_mode: 'cross' | 'isolated';
  position_size_percent: number;
  confidence_threshold: number;
  min_models_required: number;
  total_models_available: number;
  take_profit_percentage: number;
  stop_loss_percentage: number;
}

export const StrategyConfig: React.FC = () => {
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [editingStrategy, setEditingStrategy] = useState<EditStrategyData | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const itemsPerPage = 50;
  const queryClient = useQueryClient();

  const { data: strategiesData, isLoading } = useQuery({
    queryKey: ['strategy-config', currentPage, searchTerm],
    queryFn: async () => {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000);
      
      try {
        // Build query parameters
        const params = new URLSearchParams({
          page: currentPage.toString(),
          limit: itemsPerPage.toString(),
          ...(searchTerm && { search: searchTerm })
        });
        
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/strategies/paginated?${params}`,
          { signal: controller.signal }
        );
        clearTimeout(timeoutId);
        
        if (!response.ok) throw new Error('Failed to fetch strategies');
        
        const data = await response.json();
        setTotalPages(Math.ceil(data.total / itemsPerPage));
        return data;
      } catch (error) {
        clearTimeout(timeoutId);
        console.error('Failed to load strategy config:', error);
        return { strategies: [], total: 0, page: 1, pages: 1 };
      }
    },
    refetchInterval: 60000,
    retry: 2
  });

  const strategies = strategiesData?.strategies || [];
  const totalStrategies = strategiesData?.total || 0;

  const updateStrategyMutation = useMutation({
    mutationFn: async (strategy: EditStrategyData) => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/strategy/update`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(strategy)
      });
      if (!response.ok) throw new Error('Failed to update strategy');
      return response.json();
    },
    onSuccess: () => {
      toast.success('Strategy updated successfully');
      setEditDialogOpen(false);
      setEditingStrategy(null);
      queryClient.invalidateQueries({ queryKey: ['strategy-config'] });
    },
    onError: () => {
      toast.error('Failed to update strategy');
    }
  });

  const handleEditStrategy = (strategy: CoinStrategy) => {
    setEditingStrategy({
      coin_symbol: strategy.coin_symbol,
      leverage: strategy.leverage,
      margin_mode: strategy.margin_mode,
      position_size_percent: strategy.position_size_percent,
      confidence_threshold: strategy.confidence_threshold,
      min_models_required: strategy.min_models_required,
      total_models_available: strategy.total_models_available,
      take_profit_percentage: strategy.take_profit_percentage,
      stop_loss_percentage: strategy.stop_loss_percentage
    });
    setEditDialogOpen(true);
  };

  const handleSaveStrategy = () => {
    if (editingStrategy) {
      updateStrategyMutation.mutate(editingStrategy);
    }
  };

  const getMarginModeColor = (mode: string) => {
    return mode === 'cross' ? 'success' : 'warning';
  };

  if (isLoading) {
    return (
      <Box sx={{ p: 3 }}>
        <LinearProgress />
        <Typography sx={{ mt: 2 }}>Loading strategy configurations...</Typography>
      </Box>
    );
  }

  const handleSearch = (value: string) => {
    setSearchTerm(value);
    setCurrentPage(1); // Reset to first page when searching
  };

  const handlePageChange = (event: React.ChangeEvent<unknown>, value: number) => {
    setCurrentPage(value);
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">⚙️ Strategy Configuration</Typography>
        <Chip 
          label={`${totalStrategies} Total Coins`} 
          color="primary" 
          variant="outlined"
        />
      </Box>

      {/* Search and Filters */}
      <Box display="flex" gap={2} mb={3}>
        <TextField
          placeholder="Search coins... (e.g., BTC, ETH)"
          value={searchTerm}
          onChange={(e) => handleSearch(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search />
              </InputAdornment>
            ),
          }}
          sx={{ minWidth: 300 }}
        />
        <Box display="flex" alignItems="center" gap={1}>
          <Typography variant="body2" color="textSecondary">
            Page {currentPage} of {totalPages}
          </Typography>
          <Typography variant="body2" color="textSecondary">
            ({itemsPerPage} per page)
          </Typography>
        </Box>
      </Box>

      <Alert severity="info" sx={{ mb: 3 }}>
        <strong>Trading Criteria:</strong> Signals are generated when:
        <br />• <strong>Model Agreement:</strong> At least X models (of Y total) agree on direction (LONG/SHORT/HOLD)
        <br />• <strong>Confidence Threshold:</strong> Average confidence of agreeing models exceeds threshold %
        <br />• <strong>Example:</strong> "7/10 models" means 7 out of 10 models must agree, with average confidence ≥ 80%
      </Alert>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell><strong>Coin</strong></TableCell>
              <TableCell><strong>Leverage</strong></TableCell>
              <TableCell><strong>Margin Mode</strong></TableCell>
              <TableCell><strong>Position Size</strong></TableCell>
              <TableCell><strong>Take Profit</strong></TableCell>
              <TableCell><strong>Stop Loss</strong></TableCell>
              <TableCell><strong>Model Agreement</strong></TableCell>
              <TableCell><strong>Confidence Threshold</strong></TableCell>
              <TableCell><strong>Status</strong></TableCell>
              <TableCell><strong>Actions</strong></TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {strategies.map((strategy: CoinStrategy) => (
              <TableRow key={strategy.coin_symbol}>
                <TableCell>
                  <Typography variant="subtitle2" fontWeight="bold">
                    {strategy.coin_symbol}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Chip 
                    label={`${strategy.leverage}x`} 
                    color="primary" 
                    size="small"
                  />
                </TableCell>
                <TableCell>
                  <Chip 
                    label={strategy.margin_mode.toUpperCase()} 
                    color={getMarginModeColor(strategy.margin_mode) as any}
                    size="small"
                  />
                </TableCell>
                <TableCell>
                  <Typography variant="body2">
                    {strategy.position_size_percent}%
                  </Typography>
                </TableCell>
                <TableCell>
                  <Typography variant="body2" color="success.main">
                    {strategy.take_profit_percentage}%
                  </Typography>
                </TableCell>
                <TableCell>
                  <Typography variant="body2" color="error.main">
                    {strategy.stop_loss_percentage}%
                  </Typography>
                </TableCell>
                <TableCell>
                  <Chip 
                    label={`${strategy.min_models_required}/${strategy.total_models_available}`}
                    color="info"
                    size="small"
                  />
                </TableCell>
                <TableCell>
                  <Typography variant="body2">
                    {strategy.confidence_threshold}%
                  </Typography>
                </TableCell>
                <TableCell>
                  <Chip 
                    label={strategy.is_active ? 'Active' : 'Inactive'} 
                    color={strategy.is_active ? 'success' : 'default'}
                    size="small"
                  />
                </TableCell>
                <TableCell>
                  <IconButton 
                    size="small" 
                    onClick={() => handleEditStrategy(strategy)}
                    color="primary"
                  >
                    <Edit />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Pagination */}
      <Box display="flex" justifyContent="center" mt={3}>
        <Pagination
          count={totalPages}
          page={currentPage}
          onChange={handlePageChange}
          color="primary"
          size="large"
          showFirstButton
          showLastButton
        />
      </Box>

      {/* Edit Strategy Dialog */}
      <Dialog open={editDialogOpen} onClose={() => setEditDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Edit Strategy - {editingStrategy?.coin_symbol}</DialogTitle>
        <DialogContent>
          {editingStrategy && (
            <Box sx={{ pt: 2, display: 'flex', flexDirection: 'column', gap: 3 }}>
              <TextField
                label="Leverage"
                type="number"
                value={editingStrategy.leverage}
                onChange={(e) => setEditingStrategy({
                  ...editingStrategy,
                  leverage: Number(e.target.value)
                })}
                inputProps={{ min: 1, max: 125 }}
                fullWidth
              />
              
              <FormControl fullWidth>
                <InputLabel>Margin Mode</InputLabel>
                <Select
                  value={editingStrategy.margin_mode}
                  onChange={(e) => setEditingStrategy({
                    ...editingStrategy,
                    margin_mode: e.target.value as 'cross' | 'isolated'
                  })}
                >
                  <MenuItem value="cross">Cross Margin</MenuItem>
                  <MenuItem value="isolated">Isolated Margin</MenuItem>
                </Select>
              </FormControl>

              <TextField
                label="Position Size (%)"
                type="number"
                value={editingStrategy.position_size_percent}
                onChange={(e) => setEditingStrategy({
                  ...editingStrategy,
                  position_size_percent: Number(e.target.value)
                })}
                inputProps={{ min: 0.1, max: 100, step: 0.1 }}
                fullWidth
              />

              <Box sx={{ display: 'flex', gap: 2 }}>
                <TextField
                  label="Take Profit (%)"
                  type="number"
                  value={editingStrategy.take_profit_percentage}
                  onChange={(e) => setEditingStrategy({
                    ...editingStrategy,
                    take_profit_percentage: Number(e.target.value)
                  })}
                  inputProps={{ min: 0.1, max: 50, step: 0.1 }}
                  sx={{ flex: 1 }}
                />
                <TextField
                  label="Stop Loss (%)"
                  type="number"
                  value={editingStrategy.stop_loss_percentage}
                  onChange={(e) => setEditingStrategy({
                    ...editingStrategy,
                    stop_loss_percentage: Number(e.target.value)
                  })}
                  inputProps={{ min: 0.1, max: 20, step: 0.1 }}
                  sx={{ flex: 1 }}
                />
              </Box>

              <Box sx={{ display: 'flex', gap: 2 }}>
                <TextField
                  label="Min Models Required"
                  type="number"
                  value={editingStrategy.min_models_required}
                  onChange={(e) => setEditingStrategy({
                    ...editingStrategy,
                    min_models_required: Number(e.target.value)
                  })}
                  inputProps={{ min: 3, max: editingStrategy.total_models_available, step: 1 }}
                  sx={{ flex: 1 }}
                />
                <TextField
                  label="Total Models Available"
                  type="number"
                  value={editingStrategy.total_models_available}
                  onChange={(e) => setEditingStrategy({
                    ...editingStrategy,
                    total_models_available: Number(e.target.value)
                  })}
                  inputProps={{ min: 3, max: 10, step: 1 }}
                  sx={{ flex: 1 }}
                />
              </Box>

              <TextField
                label="Confidence Threshold (%)"
                type="number"
                value={editingStrategy.confidence_threshold}
                onChange={(e) => setEditingStrategy({
                  ...editingStrategy,
                  confidence_threshold: Number(e.target.value)
                })}
                inputProps={{ min: 50, max: 95, step: 1 }}
                fullWidth
              />
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialogOpen(false)}>Cancel</Button>
          <Button 
            variant="contained" 
            onClick={handleSaveStrategy}
            disabled={updateStrategyMutation.isPending}
          >
            Save Changes
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};