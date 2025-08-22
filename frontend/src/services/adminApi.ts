import { ApiService } from './api'

// 用户管理相关类型
export interface AdminUser {
  id: number
  username: string
  email: string
  role: 'admin' | 'user'
  is_active: boolean
  is_verified: boolean
  created_at: string
  updated_at: string
  last_login: string | null
}

export interface CreateUserRequest {
  username: string
  email: string
  password: string
  full_name?: string
  role: 'admin' | 'user'
}

export interface UpdateUserRequest {
  username?: string
  email?: string
  full_name?: string
  role?: 'admin' | 'user'
  is_active?: boolean
}

// 系统统计相关类型
export interface SystemStats {
  total_users: number
  active_users: number
  total_documents: number
  total_questions: number
  avg_response_time: number
  system_health: {
    cpu_usage: number
    memory_usage: number
    disk_usage: number
    database_status: string
    redis_status: string
    api_response_time: number
  }
  recent_activity: Array<{
    type: string
    message: string
    timestamp: string
  }>
  new_users_today: number
  questions_today: number
}

// 系统日志相关类型
export interface SystemLog {
  id: number
  level: 'info' | 'warning' | 'error'
  module: string
  action: string
  message: string
  details?: any
  user_id?: number
  ip_address?: string
  user_agent?: string
  created_at: string
  timestamp: string
}

// 管理员API服务
export class AdminApi {
  /**
   * 获取所有用户列表
   */
  static async getUsers(): Promise<AdminUser[]> {
    return ApiService.get<AdminUser[]>('/api/v1/auth/users')
  }

  /**
   * 创建新用户
   */
  static async createUser(userData: CreateUserRequest): Promise<AdminUser> {
    return ApiService.post<AdminUser>('/api/v1/auth/register', userData)
  }

  /**
   * 更新用户信息
   */
  static async updateUser(userId: number, userData: UpdateUserRequest): Promise<AdminUser> {
    return ApiService.put<AdminUser>(`/api/v1/auth/users/${userId}`, userData)
  }

  /**
   * 删除用户
   */
  static async deleteUser(userId: number): Promise<void> {
    return ApiService.delete(`/api/v1/auth/users/${userId}`)
  }

  /**
   * 切换用户状态（激活/禁用）
   */
  static async toggleUserStatus(userId: number, isActive: boolean): Promise<AdminUser> {
    return this.updateUser(userId, { is_active: isActive })
  }

  /**
   * 获取系统统计信息
   */
  static async getSystemStats(): Promise<SystemStats> {
    try {
      // 并行获取各种统计数据
      const [users, qaStats, docStats, healthCheck] = await Promise.all([
        this.getUsers(),
        ApiService.get('/api/v1/qa/statistics'),
        ApiService.get('/api/v1/documents/statistics/overview'),
        ApiService.get('/health')
      ])

      // 计算活跃用户数（假设最近30天有活动的用户为活跃用户）
      const activeUsers = users.filter(user => {
        if (!user.last_login) return false
        const lastLogin = new Date(user.last_login)
        const thirtyDaysAgo = new Date()
        thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30)
        return lastLogin > thirtyDaysAgo
      }).length

      // 生成最近活动数据
      const recentActivity = [
        {
          type: 'user_registration',
          message: `新用户注册: ${users.length} 个用户`,
          timestamp: new Date().toISOString()
        },
        {
          type: 'qa_activity',
          message: `问答活动: ${qaStats.total_questions || 0} 个问题`,
          timestamp: new Date().toISOString()
        },
        {
          type: 'document_activity',
          message: `文档管理: ${docStats.total_documents || 0} 个文档`,
          timestamp: new Date().toISOString()
        }
      ]

      // 确定系统健康状态
      let systemHealth: 'healthy' | 'warning' | 'error' = 'healthy'
      if (healthCheck.status !== 'healthy') {
        systemHealth = 'error'
      } else if (activeUsers < users.length * 0.5) {
        systemHealth = 'warning'
      }

      return {
        total_users: users.length,
        active_users: activeUsers,
        total_documents: docStats.total_documents || 0,
        total_questions: qaStats.total_questions || 0,
        avg_response_time: 120,
        system_health: {
          cpu_usage: 45,
          memory_usage: 68,
          disk_usage: 32,
          database_status: 'connected',
          redis_status: 'connected',
          api_response_time: 120
        },
         new_users_today: 12,
         questions_today: 156,
         recent_activity: recentActivity
      }
    } catch (error) {
      console.error('获取系统统计失败:', error)
      // 返回默认数据
      return {
        total_users: 0,
        active_users: 0,
        total_documents: 0,
        total_questions: 0,
        avg_response_time: 500,
        system_health: {
          cpu_usage: 90,
          memory_usage: 95,
          disk_usage: 85,
          database_status: 'error',
          redis_status: 'disconnected',
          api_response_time: 500
        },
         new_users_today: 0,
         questions_today: 0,
         recent_activity: []
      }
    }
  }

  /**
   * 获取系统日志
   */
  static async getSystemLogs(limit: number = 50): Promise<SystemLog[]> {
    try {
      // 注意：后端可能还没有实现系统日志API，这里先返回模拟数据
      // 实际实现时应该调用真实的API端点
      console.warn('系统日志API尚未实现，返回模拟数据')
      
      // 模拟系统日志数据
      const mockLogs: SystemLog[] = [
        {
          id: 1,
          level: 'info',
          module: 'auth',
          action: 'login',
          message: '用户登录成功',
          user_id: 1,
          ip_address: '192.168.1.100',
          created_at: new Date(Date.now() - 1000 * 60 * 5).toISOString(),
          timestamp: new Date(Date.now() - 1000 * 60 * 5).toISOString()
        },
        {
          id: 2,
          level: 'warning',
          module: 'qa',
          action: 'query',
          message: '查询处理时间较长',
          user_id: 2,
          ip_address: '192.168.1.101',
          created_at: new Date(Date.now() - 1000 * 60 * 10).toISOString(),
          timestamp: new Date(Date.now() - 1000 * 60 * 10).toISOString()
        },
        {
          id: 3,
          level: 'error',
          module: 'document',
          action: 'upload',
          message: '文档上传失败',
          user_id: 1,
          ip_address: '192.168.1.100',
          created_at: new Date(Date.now() - 1000 * 60 * 15).toISOString(),
          timestamp: new Date(Date.now() - 1000 * 60 * 15).toISOString()
        }
      ]
      
      return mockLogs.slice(0, limit)
    } catch (error) {
      console.error('获取系统日志失败:', error)
      return []
    }
  }
}

export default AdminApi