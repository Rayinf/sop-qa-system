import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios'
import { ApiResponse, ApiError } from '../types/index'

// API基础配置
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const API_TIMEOUT = 600000 // 10分钟超时，适应大文件处理时间

// 创建axios实例
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
})

// 请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    // 添加认证token
    const token = localStorage.getItem('auth-token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    
    // 添加请求ID用于追踪
    config.headers['X-Request-ID'] = generateRequestId()
    
    console.log(`[API Request] ${config.method?.toUpperCase()} ${config.url}`, {
      params: config.params,
      data: config.data
    })
    
    return config
  },
  (error) => {
    console.error('[API Request Error]', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    console.log(`[API Response] ${response.config.method?.toUpperCase()} ${response.config.url}`, {
      status: response.status,
      data: response.data
    })
    
    return response
  },
  async (error) => {
    const originalRequest = error.config
    
    console.error(`[API Error] ${originalRequest?.method?.toUpperCase()} ${originalRequest?.url}`, {
      status: error.response?.status,
      data: error.response?.data,
      message: error.message
    })
    
    // 处理401未授权错误
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true
      
      try {
        // 尝试刷新token
        const refreshToken = localStorage.getItem('auth-refresh-token')
        if (refreshToken) {
          const response = await axios.post(`${API_BASE_URL}/api/v1/auth/refresh`, {
            refresh_token: refreshToken
          })
          
          const { access_token } = response.data
          localStorage.setItem('auth-token', access_token)
          
          // 重试原始请求
          originalRequest.headers.Authorization = `Bearer ${access_token}`
          return apiClient(originalRequest)
        }
      } catch (refreshError) {
        // 刷新失败，清除认证信息并跳转到登录页
        localStorage.removeItem('auth-token')
        localStorage.removeItem('auth-refresh-token')
        window.location.href = '/login'
        return Promise.reject(refreshError)
      }
    }
    
    // 处理网络错误
    if (!error.response) {
      const networkError: ApiError = {
        code: 0,
        message: '网络连接失败，请检查网络设置',
        type: 'network_error'
      }
      return Promise.reject(networkError)
    }
    
    // 处理服务器错误
    const apiError: ApiError = {
      code: error.response.status,
      message: error.response.data?.error?.message || error.response.data?.message || '服务器错误',
      type: error.response.data?.error?.type || 'server_error',
      details: error.response.data?.error?.details
    }
    
    return Promise.reject(apiError)
  }
)

// 生成请求ID
function generateRequestId(): string {
  return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
}

// API方法封装
export class ApiService {
  // GET请求
  static async get<T = any>(
    url: string,
    config?: AxiosRequestConfig
  ): Promise<T> {
    const response = await apiClient.get<T>(url, config)
    return response.data
  }

  // POST请求
  static async post<T = any>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig
  ): Promise<T> {
    const response = await apiClient.post<T>(url, data, config)
    return response.data
  }

  // PUT请求
  static async put<T = any>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig
  ): Promise<T> {
    const response = await apiClient.put<T>(url, data, config)
    return response.data
  }

  // PATCH请求
  static async patch<T = any>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig
  ): Promise<T> {
    const response = await apiClient.patch<T>(url, data, config)
    return response.data
  }

  // DELETE请求
  static async delete<T = any>(
    url: string,
    config?: AxiosRequestConfig
  ): Promise<T> {
    const response = await apiClient.delete<T>(url, config)
    return response.data
  }

  // 文件上传
  static async upload<T = any>(
    url: string,
    file: File,
    data?: Record<string, any>,
    onProgress?: (progress: number) => void
  ): Promise<T> {
    const formData = new FormData()
    formData.append('file', file)
    
    // 添加额外数据
    if (data) {
      Object.keys(data).forEach(key => {
        formData.append(key, data[key])
      })
    }
    
    const response = await apiClient.post<T>(url, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          )
          onProgress(progress)
        }
      },
    })
    
    return response.data
  }

  // 下载文件
  static async download(
    url: string,
    filename?: string,
    config?: AxiosRequestConfig
  ): Promise<void> {
    const response = await apiClient.get(url, {
      ...config,
      responseType: 'blob',
    })
    
    // 创建下载链接
    const blob = new Blob([response.data])
    const downloadUrl = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = downloadUrl
    link.download = filename || 'download'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(downloadUrl)
  }

  // 设置认证token
  static setAuthToken(token: string): void {
    localStorage.setItem('auth-token', token)
    apiClient.defaults.headers.common['Authorization'] = `Bearer ${token}`
  }

  // 清除认证token
  static clearAuthToken(): void {
    localStorage.removeItem('auth-token')
    localStorage.removeItem('auth-refresh-token')
    delete apiClient.defaults.headers.common['Authorization']
  }

  // 获取当前token
  static getAuthToken(): string | null {
    return localStorage.getItem('auth-token')
  }

  // 检查是否已认证
  static isAuthenticated(): boolean {
    return !!this.getAuthToken()
  }
}

// 导出axios实例供特殊用途
export { apiClient }
export default ApiService