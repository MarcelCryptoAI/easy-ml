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
  LinearProgress,
  ToggleButton,
  ToggleButtonGroup,
  Divider
} from '@mui/material';
import { Edit, Settings, Search, ViewList, ViewModule, EditNote } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import toast from 'react-hot-toast';

interface CoinStrategy {
  coin_symbol: string;
  leverage: number;
  margin_mode: 'cross' | 'isolated';
  position_size_percent: number;
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
  take_profit_percentage: number;
  stop_loss_percentage: number;
}

export const StrategyConfig: React.FC = () => {
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [editingStrategy, setEditingStrategy] = useState<EditStrategyData | null>(null);
  const [bulkEditDialogOpen, setBulkEditDialogOpen] = useState(false);
  const [bulkEditData, setBulkEditData] = useState<Omit<EditStrategyData, 'coin_symbol'>>({
    leverage: 10,
    margin_mode: 'cross',
    position_size_percent: 2.0,
    take_profit_percentage: 2.0,
    stop_loss_percentage: 1.0
  });
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [showAllCoins, setShowAllCoins] = useState(true); // Default to show all coins
  const itemsPerPage = 50;
  const queryClient = useQueryClient();

  const { data: strategiesData, isLoading } = useQuery({
    queryKey: ['strategy-config', currentPage, searchTerm, showAllCoins],
    queryFn: async () => {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000);
      
      try {
        let url;
        if (showAllCoins) {
          // Use the new /strategies/all endpoint to get all coins
          url = `${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/strategies/all`;
        } else {
          // Use paginated endpoint for performance if needed
          const params = new URLSearchParams({
            page: currentPage.toString(),
            limit: itemsPerPage.toString(),
            ...(searchTerm && { search: searchTerm })
          });
          url = `${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/strategies/paginated?${params}`;
        }
        
        const response = await fetch(url, { signal: controller.signal });
        clearTimeout(timeoutId);
        
        if (!response.ok) throw new Error('Failed to fetch strategies');
        
        const data = await response.json();
        
        if (showAllCoins) {
          // Filter client-side if search term is provided
          let filteredStrategies = data.strategies;
          if (searchTerm) {
            filteredStrategies = data.strategies.filter((strategy: CoinStrategy) =>
              strategy.coin_symbol.toLowerCase().includes(searchTerm.toLowerCase())
            );
          }
          
          // Calculate pagination for filtered results
          const total = filteredStrategies.length;
          const pages = Math.ceil(total / itemsPerPage);
          setTotalPages(pages);
          
          // Apply pagination client-side
          const start = (currentPage - 1) * itemsPerPage;
          const end = start + itemsPerPage;
          const paginatedStrategies = filteredStrategies.slice(start, end);
          
          return {
            strategies: paginatedStrategies,
            total: total,
            page: currentPage,
            pages: pages
          };
        } else {
          setTotalPages(Math.ceil(data.total / itemsPerPage));
          return data;
        }
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

  const bulkUpdateStrategiesMutation = useMutation({
    mutationFn: async (data: Omit<EditStrategyData, 'coin_symbol'>) => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/strategies/bulk-update`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      if (!response.ok) throw new Error('Failed to bulk update strategies');
      return response.json();
    },
    onSuccess: (data) => {
      toast.success(`Successfully updated ${data.updated_count} strategies`);
      setBulkEditDialogOpen(false);
      queryClient.invalidateQueries({ queryKey: ['strategy-config'] });
    },
    onError: () => {
      toast.error('Failed to bulk update strategies');
    }
  });

  const handleEditStrategy = (strategy: CoinStrategy) => {
    setEditingStrategy({
      coin_symbol: strategy.coin_symbol,
      leverage: strategy.leverage,
      margin_mode: strategy.margin_mode,
      position_size_percent: strategy.position_size_percent,
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

  const handleBulkSave = () => {
    bulkUpdateStrategiesMutation.mutate(bulkEditData);
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

  const handleViewModeChange = (event: React.MouseEvent<HTMLElement>, newMode: boolean | null) => {
    if (newMode !== null) {
      setShowAllCoins(newMode);
      setCurrentPage(1); // Reset to first page when changing view mode
    }
  };

  const handlePageChange = (event: React.ChangeEvent<unknown>, value: number) => {
    setCurrentPage(value);
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">⚙️ Strategy Configuration</Typography>
        <Box display="flex" alignItems="center" gap={2}>
          <Button
            variant="contained"
            color="primary"
            startIcon={<EditNote />}
            onClick={() => setBulkEditDialogOpen(true)}
          >
            Bulk Edit All
          </Button>
          <Chip 
            label={`${totalStrategies} Total Coins`} 
            color="primary" 
            variant="outlined"
          />
        </Box>
      </Box>

      {/* Search and Filters */}
      <Box display="flex" gap={2} mb={3} alignItems="center" flexWrap="wrap">
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
        
        <ToggleButtonGroup
          value={showAllCoins}
          exclusive
          onChange={handleViewModeChange}
          aria-label="view mode"
          size="small"
        >
          <ToggleButton value={true} aria-label="show all coins">
            <ViewModule />
            <Typography sx={{ ml: 1 }}>All Coins</Typography>
          </ToggleButton>
          <ToggleButton value={false} aria-label="paginated view">
            <ViewList />
            <Typography sx={{ ml: 1 }}>Paginated</Typography>
          </ToggleButton>
        </ToggleButtonGroup>
        
        <Box display="flex" alignItems="center" gap={1}>
          <Typography variant="body2" color="textSecondary">
            Page {currentPage} of {totalPages}
          </Typography>
          <Typography variant="body2" color="textSecondary">
            ({itemsPerPage} per page)
          </Typography>
        </Box>
      </Box>

      <Alert severity="success" sx={{ mb: 3 }}>
        <strong>🤖 AI-Powered Trading Logic:</strong> Our advanced AI system automatically determines optimal trading signals by:
        <br />• <strong>Weighted Model Consensus:</strong> Each model type has different weights based on performance (Transformer: 1.25x, LSTM: 1.2x, XGBoost: 1.15x, etc.)
        <br />• <strong>Confidence-Based Scoring:</strong> Combines model weights × confidence scores × vote margins for intelligent decision making
        <br />• <strong>Automatic Thresholds:</strong> AI only trades when confidence ≥ 75% AND decision margin ≥ 0.3 (no manual tuning needed)
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
              <TableCell><strong>AI Decision Logic</strong></TableCell>
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
                    label="🤖 AI Auto-Pilot"
                    color="success"
                    size="small"
                    icon={<Settings />}
                  />
                  <Typography variant="caption" display="block" color="textSecondary">
                    Weighted consensus + confidence analysis
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

              <Alert severity="info" sx={{ mt: 2 }}>
                <strong>🤖 AI Trading Logic</strong><br />
                The system automatically uses weighted model consensus with intelligent thresholds. 
                No manual configuration needed - AI handles optimal decision making!
              </Alert>
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

      {/* Bulk Edit Dialog */}
      <Dialog open={bulkEditDialogOpen} onClose={() => setBulkEditDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>
          Bulk Edit All Strategies
          <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
            This will update all {totalStrategies} coins with the same settings
          </Typography>
        </DialogTitle>
        <DialogContent>
          <Alert severity="warning" sx={{ mb: 3 }}>
            <strong>Warning:</strong> This will apply these settings to ALL coins. Individual coin settings will be overwritten.
          </Alert>
          
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            <TextField
              label="Leverage"
              type="number"
              value={bulkEditData.leverage}
              onChange={(e) => setBulkEditData({
                ...bulkEditData,
                leverage: Number(e.target.value)
              })}
              inputProps={{ min: 1, max: 125 }}
              fullWidth
            />
            
            <FormControl fullWidth>
              <InputLabel>Margin Mode</InputLabel>
              <Select
                value={bulkEditData.margin_mode}
                onChange={(e) => setBulkEditData({
                  ...bulkEditData,
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
              value={bulkEditData.position_size_percent}
              onChange={(e) => setBulkEditData({
                ...bulkEditData,
                position_size_percent: Number(e.target.value)
              })}
              inputProps={{ min: 0.1, max: 100, step: 0.1 }}
              fullWidth
            />

            <Divider />

            <Box sx={{ display: 'flex', gap: 2 }}>
              <TextField
                label="Take Profit (%)"
                type="number"
                value={bulkEditData.take_profit_percentage}
                onChange={(e) => setBulkEditData({
                  ...bulkEditData,
                  take_profit_percentage: Number(e.target.value)
                })}
                inputProps={{ min: 0.1, max: 50, step: 0.1 }}
                sx={{ flex: 1 }}
              />
              <TextField
                label="Stop Loss (%)"
                type="number"
                value={bulkEditData.stop_loss_percentage}
                onChange={(e) => setBulkEditData({
                  ...bulkEditData,
                  stop_loss_percentage: Number(e.target.value)
                })}
                inputProps={{ min: 0.1, max: 20, step: 0.1 }}
                sx={{ flex: 1 }}
              />
            </Box>

            <Divider />

            <Alert severity="success" sx={{ mt: 2 }}>
              <strong>🤖 Intelligent AI Decision Making</strong><br />
              Our advanced AI system automatically handles model consensus and confidence thresholds using:
              <br />• Weighted model performance (Transformer: 1.25x, LSTM: 1.2x, XGBoost: 1.15x)
              <br />• Dynamic confidence scoring based on agreement strength
              <br />• Automatic risk management with 75% confidence + 0.3 margin thresholds
            </Alert>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setBulkEditDialogOpen(false)}>Cancel</Button>
          <Button 
            variant="contained" 
            color="warning"
            onClick={handleBulkSave}
            disabled={bulkUpdateStrategiesMutation.isPending}
          >
            Apply to All {totalStrategies} Coins
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};