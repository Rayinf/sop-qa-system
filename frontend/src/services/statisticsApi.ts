import { QAStatistics, DocumentStatistics, PersonalQAStatistics, QALog, PaginatedResponse } from '../types/index'
import { ApiService } from './api'

// 统计API服务
export class StatisticsApi {
  /**
   * 获取问答统计数据
   */
  static async getQAStatistics(): Promise<QAStatistics> {
    return ApiService.get<QAStatistics>('/api/v1/qa/statistics')
  }

  /**
   * 获取个人问答统计数据
   */
  static async getPersonalQAStatistics(): Promise<PersonalQAStatistics> {
    return ApiService.get<PersonalQAStatistics>('/api/v1/qa/statistics/personal')
  }

  /**
   * 获取文档统计数据
   */
  static async getDocumentStatistics(): Promise<DocumentStatistics> {
    return ApiService.get<DocumentStatistics>('/api/v1/documents/statistics/overview')
  }

  /**
   * 获取最近的问答记录
   */
  static async getRecentQAHistory(limit: number = 5): Promise<QALog[]> {
    const response = await ApiService.get<PaginatedResponse<QALog>>(`/api/v1/qa/history?limit=${limit}`)
    return response.items || []
  }

  /**
   * 获取综合统计数据（用于Dashboard）
   */
  static async getDashboardStatistics(): Promise<{
    qaStats: QAStatistics
    documentStats: DocumentStatistics
    personalStats: PersonalQAStatistics
    recentQA: QALog[]
  }> {
    const [qaStats, documentStats, personalStats, recentQA] = await Promise.all([
      this.getQAStatistics(),
      this.getDocumentStatistics(),
      this.getPersonalQAStatistics(),
      this.getRecentQAHistory(5)
    ])

    return {
      qaStats,
      documentStats,
      personalStats,
      recentQA
    }
  }
}

export default StatisticsApi