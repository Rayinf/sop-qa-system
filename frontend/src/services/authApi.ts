import { User, LoginRequest, LoginResponse, UserCreate, UserUpdate } from '../types/index'
import ApiService from './api'

// 认证API服务
class AuthApiService {
  private baseUrl = '/api/v1/auth'

  // 设置认证token
  setAuthToken(token: string): void {
    ApiService.setAuthToken(token)
  }

  // 清除认证token
  clearAuthToken(): void {
    ApiService.clearAuthToken()
  }

  // 用户登录
  async login(credentials: LoginRequest): Promise<LoginResponse> {
    // OAuth2PasswordRequestForm 需要 application/x-www-form-urlencoded
    const params = new URLSearchParams()
    params.append('username', credentials.email)
    params.append('password', credentials.password)

    const response = await ApiService.post<LoginResponse>(
      `${this.baseUrl}/login`,
      params,
      {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        }
      }
    )

    // 保存refresh token
    localStorage.setItem('auth-refresh-token', response.refresh_token)

    return response
  }

  // 用户注册
  async register(userData: UserCreate): Promise<User> {
    return ApiService.post<User>(
      `${this.baseUrl}/register`,
      userData
    )
  }

  // 用户登出
  async logout(): Promise<void> {
    await ApiService.post<void>(
      `${this.baseUrl}/logout`
    )
  }

  // 刷新token
  async refreshToken(refreshToken: string): Promise<{ access_token: string; refresh_token: string }> {
    const response = await ApiService.post<{ access_token: string; refresh_token: string }>(
      `${this.baseUrl}/refresh`,
      { refresh_token: refreshToken }
    )
    
    // 更新存储的refresh token
    localStorage.setItem('auth-refresh-token', response.refresh_token)
    
    return response
  }

  // 获取当前用户信息
  async getCurrentUser(): Promise<User> {
    return ApiService.get<User>(
      `${this.baseUrl}/me`
    )
  }

  // 更新当前用户信息
  async updateCurrentUser(userData: UserUpdate): Promise<User> {
    return ApiService.put<User>(
      `${this.baseUrl}/me`,
      userData
    )
  }

  // 修改密码
  async changePassword(oldPassword: string, newPassword: string): Promise<void> {
    await ApiService.post<void>(
      `${this.baseUrl}/change-password`,
      {
        old_password: oldPassword,
        new_password: newPassword
      }
    )
  }

  // 忘记密码
  async forgotPassword(email: string): Promise<void> {
    await ApiService.post<void>(
      `${this.baseUrl}/forgot-password`,
      { email }
    )
  }

  // 重置密码
  async resetPassword(token: string, newPassword: string): Promise<void> {
    await ApiService.post<void>(
      `${this.baseUrl}/reset-password`,
      {
        token,
        new_password: newPassword
      }
    )
  }

  // 验证邮箱
  async verifyEmail(token: string): Promise<void> {
    await ApiService.post<void>(
      `${this.baseUrl}/verify-email`,
      { token }
    )
  }

  // 重新发送验证邮件
  async resendVerificationEmail(): Promise<void> {
    await ApiService.post<void>(
      `${this.baseUrl}/resend-verification`
    )
  }
}

// 导出单例实例
export const authApi = new AuthApiService()
export default authApi