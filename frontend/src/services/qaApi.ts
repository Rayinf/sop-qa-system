import { 
  QuestionRequest, 
  AnswerResponse, 
  QALog, 
  FeedbackRequest, 
  QAStatistics,
  PaginatedResponse
} from '../types/index'
import { ApiService } from './api'

// 问答API服务
class QAApiService {
  private baseUrl = '/api/v1/qa'

  // 提问
  async askQuestion(request: QuestionRequest): Promise<AnswerResponse> {
    return ApiService.post<AnswerResponse>(
      `${this.baseUrl}/ask`,
      request
    )
  }

  // 获取问答历史（当前用户）
  async getQAHistory(sessionId?: string): Promise<PaginatedResponse<QALog>> {
    const url = sessionId 
      ? `${this.baseUrl}/history?session_id=${sessionId}`
      : `${this.baseUrl}/history`
    
    return ApiService.get<PaginatedResponse<QALog>>(url)
  }

  // 获取分页问答历史
  async getQAHistoryPaginated(
    page: number = 1,
    size: number = 20,
    sessionId?: string
  ): Promise<PaginatedResponse<QALog>> {
    const params = new URLSearchParams({
      page: page.toString(),
      size: size.toString()
    })
    
    if (sessionId) {
      params.append('session_id', sessionId)
    }
    
    return ApiService.get<PaginatedResponse<QALog>>(
      `${this.baseUrl}/history/paginated?${params.toString()}`
    )
  }

  // 删除问答历史记录
  async deleteQAHistory(qaLogId: string): Promise<void> {
    await ApiService.delete<void>(
      `${this.baseUrl}/history/${qaLogId}`
    )
  }

  // 提交反馈
  async submitFeedback(feedback: FeedbackRequest): Promise<void> {
    await ApiService.post<void>(
      `${this.baseUrl}/feedback`,
      feedback
    )
  }

  // 获取问答统计
  async getQAStats(): Promise<QAStatistics> {
    return ApiService.get<QAStatistics>(
      `${this.baseUrl}/stats`
    )
  }

  // 获取个人问答统计
  async getPersonalQAStats(): Promise<QAStatistics> {
    return ApiService.get<QAStatistics>(
      `${this.baseUrl}/stats/personal`
    )
  }

  // 清除问答缓存（管理员功能）
  async clearQACache(): Promise<void> {
    await ApiService.post<void>(
      `${this.baseUrl}/cache/clear`
    )
  }

  // 获取问题建议
  async getQuestionSuggestions(query?: string): Promise<string[]> {
    const url = query 
      ? `${this.baseUrl}/suggestions?query=${encodeURIComponent(query)}`
      : `${this.baseUrl}/suggestions`
    
    // 添加缓存控制头，确保获取最新数据
    const response = await ApiService.get<{
      query?: string
      total_suggestions: number
      suggestions: Array<{
        question: string
        count: number
        category: string
      }>
    }>(url, {
      headers: {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
      }
    })
    
    // 提取问题文本数组
    return response.suggestions.map(item => item.question)
  }

  // 获取向量搜索日志
  async getVectorSearchLogs(question: string, activeKbIds?: string[]): Promise<{
    logs: Array<{
      timestamp: string
      level: string
      message: string
      details: any
    }>
    status: string
    total_logs: number
  }> {
    const params = new URLSearchParams({
      question: question
    })
    if (activeKbIds && activeKbIds.length > 0) {
      params.append('active_kb_ids', activeKbIds.join(','))
    }
    return ApiService.get<{
      logs: Array<{
        timestamp: string
        level: string
        message: string
        details: any
      }>
      status: string
      total_logs: number
    }>(`${this.baseUrl}/vector-logs?${params.toString()}`)
  }

  // 批量提问（管理员功能）
  async batchAskQuestions(questions: string[]): Promise<{ task_id: string }> {
    return ApiService.post<{ task_id: string }>(
      `${this.baseUrl}/batch/ask`,
      { questions }
    )
  }

  // 健康检查
  async healthCheck(): Promise<{
    status: string
    components: {
      llm_service: string
      vector_service: string
      cache_service: string
    }
  }> {
    return ApiService.get<{
      status: string
      components: {
        llm_service: string
        vector_service: string
        cache_service: string
      }
    }>(`${this.baseUrl}/health`)
  }

  // 获取会话列表
  async getSessions(): Promise<Array<{
    session_id: string
    created_at: string
    question_count: number
    last_question: string
  }>> {
    return ApiService.get<Array<{
      session_id: string
      created_at: string
      question_count: number
      last_question: string
    }>>(`${this.baseUrl}/sessions`)
  }

  // 删除会话
  async deleteSession(sessionId: string): Promise<void> {
    await ApiService.delete<void>(
      `${this.baseUrl}/sessions/${sessionId}`
    )
  }

  // 重命名会话
  async renameSession(sessionId: string, title: string): Promise<void> {
    await ApiService.put<void>(
      `${this.baseUrl}/sessions/${sessionId}`,
      { title }
    )
  }

  // =====================
  // LLM 模型相关接口
  // =====================

  // 获取可用模型列表
  async getAvailableModels(): Promise<string[]> {
    const res = await ApiService.get<{ available_models: string[] }>(
      `${this.baseUrl}/available-models`
    )
    return res.available_models
  }

  // 获取当前使用的模型
  async getCurrentModel(): Promise<string> {
    const res = await ApiService.get<{ current_model: string }>(
      `${this.baseUrl}/current-model`
    )
    return res.current_model
  }

  // 切换LLM模型
  async switchModel(modelName: string, config?: Record<string, any>): Promise<{ success: boolean; message?: string }> {
    return ApiService.post<{ success: boolean; message?: string }>(
      `${this.baseUrl}/switch-model`,
      { model_name: modelName, config }
    )
  }

  // Kimi文件上传相关API
  async uploadFileToKimi(file: File): Promise<{
    success: boolean
    message: string
    file_info: {
      id: string
      name: string
      size: number
      type: string
      created_at: string
    }
    content_preview: string
  }> {
    const formData = new FormData()
    formData.append('file', file)
    
    return ApiService.post<{
      success: boolean
      message: string
      file_info: {
        id: string
        name: string
        size: number
        type: string
        created_at: string
      }
      content_preview: string
    }>(
      `${this.baseUrl}/kimi/upload-file`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      }
    )
  }

  // 获取Kimi文件列表
  async getKimiFiles(): Promise<{
    success: boolean
    files: Array<{
      id: string
      name: string
      size: number
      type: string
      created_at: string
    }>
    total: number
  }> {
    return ApiService.get<{
      success: boolean
      files: Array<{
        id: string
        name: string
        size: number
        type: string
        created_at: string
      }>
      total: number
    }>(`${this.baseUrl}/kimi/files`)
  }

  // 获取Kimi文件信息
  async getKimiFileInfo(fileId: string): Promise<{
    success: boolean
    file_info: {
      id: string
      name: string
      size: number
      type: string
      created_at: string
    }
  }> {
    return ApiService.get<{
      success: boolean
      file_info: {
        id: string
        name: string
        size: number
        type: string
        created_at: string
      }
    }>(`${this.baseUrl}/kimi/files/${fileId}`)
  }

  // 获取Kimi文件内容
  async getKimiFileContent(fileId: string): Promise<{
    success: boolean
    content: string
  }> {
    return ApiService.get<{
      success: boolean
      content: string
    }>(`${this.baseUrl}/kimi/files/${fileId}/content`)
  }

  // 删除Kimi文件
  async deleteKimiFile(fileId: string): Promise<{
    success: boolean
    message: string
  }> {
    return ApiService.delete<{
      success: boolean
      message: string
    }>(`${this.baseUrl}/kimi/files/${fileId}`)
  }

  // 批量上传文件到Kimi
  async batchUploadFilesToKimi(files: File[]): Promise<{
    success: boolean
    message: string
    results: Array<{
      filename: string
      success: boolean
      file_info?: {
        id: string
        name: string
        size: number
        type: string
        created_at: string
      }
      content_preview?: string
      error?: string
    }>
  }> {
    const formData = new FormData()
    files.forEach(file => {
      formData.append('files', file)
    })
    
    return ApiService.post<{
      success: boolean
      message: string
      results: Array<{
        filename: string
        success: boolean
        file_info?: {
          id: string
          name: string
          size: number
          type: string
          created_at: string
        }
        content_preview?: string
        error?: string
      }>
    }>(
      `${this.baseUrl}/kimi/batch-upload`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      }
    )
  }
}

export const qaApi = new QAApiService()
export default qaApi