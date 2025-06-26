import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Tabs,
  Tab,
  Box
} from '@mui/material';
import { Dashboard } from '../components/Dashboard';
import { CoinAnalysis } from '../components/CoinAnalysis';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && <Box>{children}</Box>}
    </div>
  );
}

export default function Home() {
  const [tabValue, setTabValue] = useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  return (
    <>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            🤖 Crypto Trading ML Platform
          </Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth={false} disableGutters>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange} aria-label="main navigation">
            <Tab label="Dashboard" />
            <Tab label="Coin Analysis" />
          </Tabs>
        </Box>

        <TabPanel value={tabValue} index={0}>
          <Dashboard />
        </TabPanel>
        
        <TabPanel value={tabValue} index={1}>
          <CoinAnalysis />
        </TabPanel>
      </Container>
    </>
  );
}