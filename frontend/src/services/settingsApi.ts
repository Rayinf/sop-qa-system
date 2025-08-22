import { ApiService } from './api'

// 系统设置相关类型
export interface SystemSettings {
  site_name: string
  site_description: string
  max_file_size: number
  allowed_file_types: string[]
  max_questions_per_day: number
  enable_registration: boolean
  enable_email_verification: boolean
  enable_file_upload: boolean
  default_user_role: string
  session_timeout: number
  api_rate_limit: number
  enable_logging: boolean
  log_level: string
  backup_enabled: boolean
  backup_frequency: string
  maintenance_mode: boolean
}

// 通知设置相关类型
export interface NotificationSettings {
  email_notifications: boolean
  browser_notifications: boolean
  new_document_alerts: boolean
  system_alerts: boolean
  weekly_reports: boolean
  notification_sound: boolean
}

// 安全设置相关类型
export interface SecuritySettings {
  password_min_length: number
  password_require_uppercase: boolean
  password_require_lowercase: boolean
  password_require_numbers: boolean
  password_require_symbols: boolean
  session_timeout_minutes: number
  max_login_attempts: number
  enable_two_factor: boolean
  ip_whitelist: string[]
}

/**
 * 设置管理API服务
 */
export class SettingsApi {
  /**
   * 获取系统设置
   */
  static async getSystemSettings(): Promise<SystemSettings> {
    try {
      // 目前后端没有设置API，使用模拟数据
      // TODO: 当后端实现设置API后，替换为真实API调用
      // return ApiService.get<SystemSettings>('/api/v1/settings/system')
      
      // 模拟API延迟
      await new Promise(resolve => setTimeout(resolve, 800))
      
      return {
          site_name: 'langchain知识库问答系统',
          site_description: '基于AI的企业知识库问答平台',
        max_file_size: 1024,
        allowed_file_types: ['pdf', 'doc', 'docx', 'txt', 'md'],
        max_questions_per_day: 100,
        enable_registration: true,
        enable_email_verification: true,
        enable_file_upload: true,
        default_user_role: 'user',
        session_timeout: 30,
        api_rate_limit: 1000,
        enable_logging: true,
        log_level: 'info',
        backup_enabled: true,
        backup_frequency: 'daily',
        maintenance_mode: false
      }
    } catch (error) {
      console.error('获取系统设置失败:', error)
      throw error
    }
  }

  /**
   * 更新系统设置
   */
  static async updateSystemSettings(settings: Partial<SystemSettings>): Promise<SystemSettings> {
    try {
      // 目前后端没有设置API，使用模拟数据
      // TODO: 当后端实现设置API后，替换为真实API调用
      // return ApiService.put<SystemSettings>('/api/v1/settings/system', settings)
      
      // 模拟API延迟
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      // 模拟返回更新后的设置
      const currentSettings = await this.getSystemSettings()
      return { ...currentSettings, ...settings }
    } catch (error) {
      console.error('更新系统设置失败:', error)
      throw error
    }
  }

  /**
   * 获取通知设置
   */
  static async getNotificationSettings(): Promise<NotificationSettings> {
    try {
      // 目前后端没有设置API，使用模拟数据
      // TODO: 当后端实现设置API后，替换为真实API调用
      // return ApiService.get<NotificationSettings>('/api/v1/settings/notifications')
      
      // 模拟API延迟
      await new Promise(resolve => setTimeout(resolve, 600))
      
      return {
        email_notifications: true,
        browser_notifications: false,
        new_document_alerts: true,
        system_alerts: true,
        weekly_reports: false,
        notification_sound: true
      }
    } catch (error) {
      console.error('获取通知设置失败:', error)
      throw error
    }
  }

  /**
   * 更新通知设置
   */
  static async updateNotificationSettings(settings: Partial<NotificationSettings>): Promise<NotificationSettings> {
    try {
      // 目前后端没有设置API，使用模拟数据
      // TODO: 当后端实现设置API后，替换为真实API调用
      // return ApiService.put<NotificationSettings>('/api/v1/settings/notifications', settings)
      
      // 模拟API延迟
      await new Promise(resolve => setTimeout(resolve, 800))
      
      // 模拟返回更新后的设置
      const currentSettings = await this.getNotificationSettings()
      return { ...currentSettings, ...settings }
    } catch (error) {
      console.error('更新通知设置失败:', error)
      throw error
    }
  }

  /**
   * 获取安全设置
   */
  static async getSecuritySettings(): Promise<SecuritySettings> {
    try {
      // 目前后端没有设置API，使用模拟数据
      // TODO: 当后端实现设置API后，替换为真实API调用
      // return ApiService.get<SecuritySettings>('/api/v1/settings/security')
      
      // 模拟API延迟
      await new Promise(resolve => setTimeout(resolve, 700))
      
      return {
        password_min_length: 8,
        password_require_uppercase: true,
        password_require_lowercase: true,
        password_require_numbers: true,
        password_require_symbols: false,
        session_timeout_minutes: 30,
        max_login_attempts: 5,
        enable_two_factor: false,
        ip_whitelist: ['192.168.1.0/24', '10.0.0.0/8']
      }
    } catch (error) {
      console.error('获取安全设置失败:', error)
      throw error
    }
  }

  /**
   * 更新安全设置
   */
  static async updateSecuritySettings(settings: Partial<SecuritySettings>): Promise<SecuritySettings> {
    try {
      // 目前后端没有设置API，使用模拟数据
      // TODO: 当后端实现设置API后，替换为真实API调用
      // return ApiService.put<SecuritySettings>('/api/v1/settings/security', settings)
      
      // 模拟API延迟
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      // 模拟返回更新后的设置
      const currentSettings = await this.getSecuritySettings()
      return { ...currentSettings, ...settings }
    } catch (error) {
      console.error('更新安全设置失败:', error)
      throw error
    }
  }

  /**
   * 重置所有设置为默认值
   */
  static async resetAllSettings(): Promise<void> {
    try {
      // 目前后端没有设置API，使用模拟数据
      // TODO: 当后端实现设置API后，替换为真实API调用
      // return ApiService.post('/api/v1/settings/reset')
      
      // 模拟API延迟
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      // 模拟重置操作成功
      console.log('所有设置已重置为默认值')
    } catch (error) {
      console.error('重置设置失败:', error)
      throw error
    }
  }

  /**
   * 导出设置配置
   */
  static async exportSettings(): Promise<Blob> {
    try {
      // 目前后端没有设置API，使用模拟数据
      // TODO: 当后端实现设置API后，替换为真实API调用
      // return ApiService.get('/api/v1/settings/export', { responseType: 'blob' })
      
      // 模拟API延迟
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      // 获取所有设置
      const [systemSettings, notificationSettings, securitySettings] = await Promise.all([
        this.getSystemSettings(),
        this.getNotificationSettings(),
        this.getSecuritySettings()
      ])
      
      // 创建配置对象
      const config = {
        system: systemSettings,
        notifications: notificationSettings,
        security: securitySettings,
        exported_at: new Date().toISOString()
      }
      
      // 转换为JSON并创建Blob
      const jsonString = JSON.stringify(config, null, 2)
      return new Blob([jsonString], { type: 'application/json' })
    } catch (error) {
      console.error('导出设置失败:', error)
      throw error
    }
  }

  /**
   * 导入设置配置
   */
  static async importSettings(file: File): Promise<void> {
    try {
      // 目前后端没有设置API，使用模拟数据
      // TODO: 当后端实现设置API后，替换为真实API调用
      
      // 读取文件内容
      const text = await file.text()
      const config = JSON.parse(text)
      
      // 验证配置格式
      if (!config.system || !config.notifications || !config.security) {
        throw new Error('配置文件格式不正确')
      }
      
      // 模拟API延迟
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      // 模拟导入操作成功
      console.log('设置配置导入成功')
    } catch (error) {
      console.error('导入设置失败:', error)
      throw error
    }
  }
}

export default SettingsApi