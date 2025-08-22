import { 
  Document, 
  DocumentCreate, 
  DocumentUpdate, 
  DocumentChunk, 
  DocumentStatistics,
  PaginatedResponse,
  SearchParams
} from '../types/index'
import { ApiService } from './api'

// 文档API服务
class DocumentApiService {
  private baseUrl = '/api/v1/documents'

  // 获取文档列表
  async getDocuments(params?: SearchParams): Promise<PaginatedResponse<Document>> {
    const queryParams = new URLSearchParams()
    if (params?.query) queryParams.append('query', params.query)
    if (params?.category) queryParams.append('category', params.category)
    if (params?.status) queryParams.append('status', params.status)
    if (params?.start_date) queryParams.append('start_date', params.start_date)
    if (params?.end_date) queryParams.append('end_date', params.end_date)
    
    const url = queryParams.toString() ? `${this.baseUrl}?${queryParams.toString()}` : this.baseUrl
    return ApiService.get<PaginatedResponse<Document>>(url)
  }

  // 获取单个文档
  async getDocument(id: number): Promise<Document> {
    return ApiService.get<Document>(
      `${this.baseUrl}/${id}`
    )
  }

  // 上传文档
  async uploadDocument(
    file: File,
    metadata?: Record<string, any>,
    onProgress?: (progress: number) => void
  ): Promise<Document> {
    // 构建查询参数
    const queryParams = new URLSearchParams()
    if (metadata) {
      Object.keys(metadata).forEach(key => {
        if (metadata[key] !== undefined && metadata[key] !== null) {
          queryParams.append(key, String(metadata[key]))
        }
      })
    }
    
    const url = `${this.baseUrl}/upload${queryParams.toString() ? '?' + queryParams.toString() : ''}`
    
    return ApiService.upload<Document>(
      url,
      file,
      undefined, // 不传递metadata到FormData
      onProgress
    )
  }

  // 更新文档
  async updateDocument(id: number, data: DocumentUpdate): Promise<Document> {
    return ApiService.put<Document>(
      `${this.baseUrl}/${id}`,
      data
    )
  }

  // 删除文档
  async deleteDocument(id: number): Promise<void> {
    await ApiService.delete<void>(
      `${this.baseUrl}/${id}`
    )
  }

  // 获取文档块
  async getDocumentChunks(id: number): Promise<DocumentChunk[]> {
    return ApiService.get<DocumentChunk[]>(
      `${this.baseUrl}/${id}/chunks`
    )
  }

  // 向量化文档
  async vectorizeDocument(id: number): Promise<{ task_id: string }> {
    return ApiService.post<{ task_id: string }>(`${this.baseUrl}/${String(id)}/vectorize`, {})
  }

  // 获取向量化进度
  async getVectorizationProgress(id: number): Promise<{
    document_id: string;
    status: string;
    progress: number;
    current_step: string;
    total_steps: number;
    current_step_index: number;
    message: string;
    error: string | null;
  }> {
    return ApiService.get<{
      document_id: string;
      status: string;
      progress: number;
      current_step: string;
      total_steps: number;
      current_step_index: number;
      message: string;
      error: string | null;
    }>(
      `${this.baseUrl}/${String(id)}/vectorize/progress`
    )
  }

  // 搜索文档
  async searchDocuments(query: string, filters?: any): Promise<any> {
    return ApiService.post<any>(
      `${this.baseUrl}/search`,
      {
        query,
        filters
      }
    )
  }

  // 获取文档统计
  async getDocumentStats(): Promise<DocumentStatistics> {
    return ApiService.get<DocumentStatistics>(`${this.baseUrl}/stats`)
  }

  // 批量操作
  async bulkOperation(
    operation: 'vectorize' | 'delete',
    documentIds: number[]
  ): Promise<{ task_id: string }> {
    return ApiService.post<{ task_id: string }>(
      `${this.baseUrl}/bulk/${operation}`,
      { document_ids: documentIds }
    )
  }

  // 重建向量索引
  async rebuildVectorIndex(): Promise<{ task_id: string }> {
    return ApiService.post<{ task_id: string }>(`${this.baseUrl}/rebuild-index`)
  }

  // 获取文档类别
  async getCategories(): Promise<string[]> {
    return ApiService.get<string[]>(`${this.baseUrl}/categories`)
  }

  // 获取文档状态
  async getStatuses(): Promise<string[]> {
    return ApiService.get<string[]>(`${this.baseUrl}/statuses`)
  }

  // 下载文档
  async downloadDocument(id: number): Promise<void> {
    return ApiService.download(`${this.baseUrl}/${id}/download`)
  }

  // 获取文档内容预览
  async getDocumentContent(id: number): Promise<{
    document_id: string
    total_chunks: number
    chunks: Array<{
      chunk_id: string
      chunk_index: number
      content: string
      page_number?: number
      metadata?: any
    }>
  }> {
    return ApiService.get<{
      document_id: string
      total_chunks: number
      chunks: Array<{
        chunk_id: string
        chunk_index: number
        content: string
        page_number?: number
        metadata?: any
      }>
    }>(`${this.baseUrl}/${id}/chunks`)
  }
}

// 导出单例实例
export const documentApi = new DocumentApiService()
export default documentApi