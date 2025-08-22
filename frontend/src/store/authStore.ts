import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { User, LoginRequest, LoginResponse } from '../types/index'
import { authApi } from '../services/authApi'

interface AuthState {
  // 状态
  user: User | null
  token: string | null
  refreshToken: string | null
  isAuthenticated: boolean
  isLoading: boolean
  error: string | null

  // 操作
  login: (credentials: LoginRequest) => Promise<void>
  logout: () => void
  refreshAccessToken: () => Promise<void>
  updateUser: (user: User) => void
  clearError: () => void
  setLoading: (loading: boolean) => void
  checkAuthStatus: () => Promise<void>
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      // 初始状态
      user: null,
      token: null,
      refreshToken: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

      // 登录
      login: async (credentials: LoginRequest) => {
        try {
          set({ isLoading: true, error: null })
          
          const response = await authApi.login(credentials)
          const { access_token, refresh_token, user } = response
          
          set({
            user,
            token: access_token,
            refreshToken: refresh_token,
            isAuthenticated: true,
            isLoading: false,
            error: null
          })
          
          // 设置API默认token
          authApi.setAuthToken(access_token)
          
        } catch (error: any) {
          set({
            user: null,
            token: null,
            refreshToken: null,
            isAuthenticated: false,
            isLoading: false,
            error: error.message || '登录失败'
          })
          throw error
        }
      },

      // 登出
      logout: () => {
        set({
          user: null,
          token: null,
          refreshToken: null,
          isAuthenticated: false,
          isLoading: false,
          error: null
        })
        
        // 清除API token
        authApi.clearAuthToken()
        
        // 调用后端登出接口
        authApi.logout().catch(console.error)
      },

      // 刷新访问令牌
      refreshAccessToken: async () => {
        try {
          const { refreshToken } = get()
          if (!refreshToken) {
            throw new Error('没有刷新令牌')
          }
          
          const response = await authApi.refreshToken(refreshToken)
          const { access_token, refresh_token } = response
          
          set({
            token: access_token,
            refreshToken: refresh_token,
            error: null
          })
          
          // 更新API token
          authApi.setAuthToken(access_token)
          
        } catch (error: any) {
          // 刷新失败，清除认证状态
          get().logout()
          throw error
        }
      },

      // 更新用户信息
      updateUser: (user: User) => {
        set({ user })
      },

      // 清除错误
      clearError: () => {
        set({ error: null })
      },

      // 设置加载状态
      setLoading: (loading: boolean) => {
        set({ isLoading: loading })
      },

      // 检查认证状态
      checkAuthStatus: async () => {
        try {
          const { token } = get()
          if (!token) {
            return
          }
          
          set({ isLoading: true })
          
          // 设置API token
          authApi.setAuthToken(token)
          
          // 获取当前用户信息
          const user = await authApi.getCurrentUser()
          
          set({
            user,
            isAuthenticated: true,
            isLoading: false,
            error: null
          })
          
        } catch (error: any) {
          // 认证失败，尝试刷新token
          try {
            await get().refreshAccessToken()
            // 重新获取用户信息
            const user = await authApi.getCurrentUser()
            set({
              user,
              isAuthenticated: true,
              isLoading: false,
              error: null
            })
          } catch (refreshError) {
            // 刷新也失败，清除认证状态
            get().logout()
          }
        }
      }
    }),
    {
      name: 'auth-storage',
      partialize: (state: any) => ({
        token: state.token,
        refreshToken: state.refreshToken,
        user: state.user
      })
    }
  )
)

// 初始化认证状态
if (typeof window !== 'undefined') {
  // 在浏览器环境中检查认证状态
  const authStore = useAuthStore.getState()
  if (authStore.token) {
    authStore.checkAuthStatus()
  }
}