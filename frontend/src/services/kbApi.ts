import { KnowledgeBase, KnowledgeBaseCreate, KnowledgeBaseUpdate, PaginatedResponse, KnowledgeBasePaginatedResponse } from '../types'
import ApiService from './api'

// 获取知识库列表
export const getKnowledgeBases = async (params?: {
  page?: number
  size?: number
  category?: string
  is_active?: boolean
}): Promise<KnowledgeBasePaginatedResponse> => {
  return await ApiService.get('/api/v1/kb', { params })
}

// 获取单个知识库详情
export const getKnowledgeBase = async (id: string): Promise<KnowledgeBase> => {
  return await ApiService.get(`/api/v1/kb/${id}`)
}

// 创建知识库
export const createKnowledgeBase = async (data: KnowledgeBaseCreate): Promise<KnowledgeBase> => {
  return await ApiService.post('/api/v1/kb', data)
}

// 更新知识库
export const updateKnowledgeBase = async (id: string, data: KnowledgeBaseUpdate): Promise<KnowledgeBase> => {
  return await ApiService.patch(`/api/v1/kb/${id}`, data)
}

// 删除知识库
export const deleteKnowledgeBase = async (id: string): Promise<void> => {
  await ApiService.delete(`/api/v1/kb/${id}`)
}

// 获取知识库的文档列表
export const getKnowledgeBaseDocuments = async (id: string, params?: {
  page?: number
  size?: number
}): Promise<PaginatedResponse<any>> => {
  return await ApiService.get(`/api/v1/kb/${id}/documents`, { params })
}

// 向知识库添加文档
export const addDocumentToKnowledgeBase = async (kbId: string, documentId: number): Promise<void> => {
  await ApiService.post(`/api/v1/kb/${kbId}/documents/${documentId}`)
}

// 从知识库移除文档
export const removeDocumentFromKnowledgeBase = async (kbId: string, documentId: number): Promise<void> => {
  await ApiService.delete(`/api/v1/kb/${kbId}/documents/${documentId}`)
}

// 获取活跃的知识库列表（用于选择器）
export const getActiveKnowledgeBases = async (): Promise<KnowledgeBase[]> => {
  const response = await ApiService.get('/api/v1/kb', { params: { is_active: true, size: 1000 } })
  return response?.kbs || []
}