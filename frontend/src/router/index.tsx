import { createBrowserRouter, Navigate } from 'react-router-dom'
import { ReactNode } from 'react'
import MainLayout from '../components/Layout/MainLayout'
import Login from '../pages/Login'
import Dashboard from '../pages/Dashboard'
import QA from '../pages/QA'
import Documents from '../pages/Documents'
import Profile from '../pages/Profile'
import Admin from '../pages/Admin'
import Settings from '../pages/Settings'

interface ProtectedRouteProps {
  children: ReactNode
  requiredRole?: string
}

interface PublicRouteProps {
  children: ReactNode
}

// 路由保护组件
const ProtectedRoute = ({ children, requiredRole }: ProtectedRouteProps): JSX.Element => {
  const token = localStorage.getItem('access_token')
  const userStr = localStorage.getItem('user')
  
  if (!token) {
    return <Navigate to="/login" replace />
  }
  
  if (requiredRole && userStr) {
    try {
      const user = JSON.parse(userStr)
      if (user.role !== requiredRole && user.role !== 'admin') {
        return <Navigate to="/dashboard" replace />
      }
    } catch (error) {
      return <Navigate to="/login" replace />
    }
  }
  
  return <>{children}</>
}

// 公共路由组件（已登录用户重定向到仪表板）
const PublicRoute = ({ children }: PublicRouteProps): JSX.Element => {
  const token = localStorage.getItem('access_token')
  
  if (token) {
    return <Navigate to="/dashboard" replace />
  }
  
  return <>{children}</>
}

const router = createBrowserRouter([
  {
    path: '/login',
    element: (
      <PublicRoute>
        <Login onLogin={async (): Promise<void> => {}} />
      </PublicRoute>
    )
  },
  {
    path: '/',
    element: (
      <ProtectedRoute>
        <MainLayout />
      </ProtectedRoute>
    ),
    children: [
      {
        index: true,
        element: <Navigate to="/dashboard" replace />
      },
      {
        path: 'dashboard',
        element: <Dashboard />
      },
      {
        path: 'qa',
        element: <QA />
      },
      {
        path: 'documents',
        element: <Documents />
      },
      {
        path: 'profile',
        element: <Profile />
      },
      {
        path: 'admin',
        element: (
          <ProtectedRoute requiredRole="admin">
            <Admin />
          </ProtectedRoute>
        )
      },
      {
        path: 'settings',
        element: (
          <ProtectedRoute requiredRole="admin">
            <Settings />
          </ProtectedRoute>
        )
      }
    ]
  },
  {
    path: '*',
    element: <Navigate to="/dashboard" replace />
  }
])

export default router