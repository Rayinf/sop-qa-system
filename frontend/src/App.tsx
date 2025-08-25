import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { Layout } from 'antd'

import { useAuthStore } from './store/authStore'
import MainLayout from './components/Layout/MainLayout'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import QA from './pages/QA'
import UnifiedKnowledgeManagement from './pages/UnifiedKnowledgeManagement'

import Profile from './pages/Profile'
import Admin from './pages/Admin'
import Settings from './pages/Settings'
import ProtectedRoute from './components/ProtectedRoute'

const App: React.FC = () => {
  const { isAuthenticated, isLoading, login } = useAuthStore()

  // 显示加载状态
  if (isLoading) {
    return (
      <Layout className="h-full flex-center">
        <div className="loading-spinner" />
        <span className="ml-2">正在加载...</span>
      </Layout>
    )
  }

  return (
    <Routes>
      {/* 公开路由 */}
      <Route
        path="/login"
        element={
          <ProtectedRoute requireAuth={false}>
            <Login onLogin={login} />
          </ProtectedRoute>
        }
      />

      {/* 受保护的路由 */}
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <MainLayout />
          </ProtectedRoute>
        }
      >
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="dashboard" element={<Dashboard />} />
        <Route path="qa" element={<QA />} />
        <Route path="knowledge" element={<UnifiedKnowledgeManagement />} />

        <Route path="profile" element={<Profile />} />
        <Route path="admin" element={<Admin />} />
        <Route path="settings" element={<Settings />} />
      </Route>
    </Routes>
  )
}

export default App