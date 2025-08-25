// 用户相关类型
export interface User {
  id: number
  email: string
  username: string
  full_name: string
  role: 'admin' | 'manager' | 'user'
  is_active: boolean
  is_verified: boolean
  created_at: string
  updated_at: string
  last_login?: string
}

export interface UserCreate {
  email: string
  username: string
  full_name: string
  password: string
  role?: 'admin' | 'manager' | 'user'
}

export interface UserUpdate {
  email?: string
  username?: string
  full_name?: string
  role?: 'admin' | 'manager' | 'user'
  is_active?: boolean
}

export interface LoginRequest {
  email: string
  password: string
}

export interface LoginResponse {
  access_token: string
  refresh_token: string
  token_type: string
  expires_in: number
  user: User
}

// 文档相关类型
export interface Document {
  id: number
  title: string
  filename: string
  file_path: string
  file_size: number
  file_type: string
  category?: string
  description?: string
  status: 'uploaded' | 'processing' | 'processed' | 'vectorizing' | 'vectorized' | 'failed' | 'error'
  is_vectorized: boolean
  chunk_count: number
  upload_user_id: number
  upload_user?: User
  created_at: string
  updated_at: string
  file_hash: string
  kb_id?: string
}

export interface DocumentCreate {
  title: string
  category?: string
  description?: string
  auto_vectorize?: boolean
}

export interface DocumentUpdate {
  title?: string
  category?: string
  description?: string
  status?: 'uploaded' | 'processing' | 'processed' | 'vectorizing' | 'vectorized' | 'failed' | 'error'
  kb_id?: string
}

export interface DocumentChunk {
  id: number
  document_id: number
  content: string
  chunk_index: number
  metadata: Record<string, any>
  created_at: string
}

// 问答相关类型
export interface QuestionRequest {
  question: string
  category?: string
  session_id?: string
  context?: string
  active_kb_ids?: string[]
  overrides?: Record<string, any>
}

export interface SourceDocument {
  document_id: number
  document_title: string
  chunk_id: string
  chunk_content: string
  similarity_score: number
  metadata: Record<string, any>
  kb_id?: string
  kb_name?: string
  page_number?: number
}

export interface FormattedAnswer {
  text: string
  confidence_score: number
  source_count: number
  processing_time: number
}

export interface AnswerResponse {
  question: string
  answer: string
  source_documents: SourceDocument[]
  confidence: number
  processing_time: number
  from_cache?: boolean
  session_id?: string
  metadata?: Record<string, any>
  formatted_answer?: FormattedAnswer
  token_usage?: Record<string, number>
  created_at?: string
}

export interface QALog {
  id: number
  question: string
  answer: string
  user_id: number
  user?: User
  session_id?: string
  source_documents: SourceDocument[]
  processing_time: number
  feedback_score?: number
  feedback_comment?: string
  created_at: string
}

// QA页面使用的类型
export interface QAItem {
  id: string
  question: string
  answer: string
  confidence_score: number
  created_at: string
  user_id: string
  feedback: number | null
  sources: Array<{
    document_id: string
    document_title: string
    chunk_id: string
    relevance_score: number
    content: string
    kb_id?: string
    kb_name?: string
    page_number?: number
  }>
  processing_time: number
  type: 'question' | 'answer'
}

export interface QAHistory {
  id: string
  question: string
  answer: string
  confidence_score: number
  created_at: string
  user_id: string
  feedback: number | null
  processing_time: number
}

export interface FeedbackRequest {
  qa_log_id: string;
  satisfaction_score: number;
  feedback?: string;
  is_helpful: boolean;
}

// 分页相关类型
export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  size: number
  pages: number
}

export interface KnowledgeBasePaginatedResponse {
  kbs: KnowledgeBase[]
  total: number
  page: number
  size: number
}

export interface PaginationParams {
  page?: number
  size?: number
  skip?: number
  limit?: number
}

// 搜索相关类型
export interface SearchParams {
  query?: string
  category?: string
  status?: string
  start_date?: string
  end_date?: string
}

// 统计相关类型
export interface QAStatistics {
  total_questions: number
  today_questions: number
  average_processing_time: number
  feedback_distribution: Record<number, number>
  popular_questions: Array<{
    question: string
    count: number
  }>
  last_updated: string
}

export interface DocumentStatistics {
  total_documents: number
  status_distribution: Record<string, number>
  category_distribution: Record<string, number>
  vector_store_stats: any
  last_updated: string
}

export interface PersonalQAStatistics {
  total_questions: number
  recent_questions_count: number
  average_processing_time: number
  feedback_distribution: Record<number, number>
  recent_questions: Array<{
    id: number
    question: string
    created_at: string
    feedback_score?: number
  }>
  last_updated: string
}

// API响应类型
export interface ApiResponse<T = any> {
  data?: T
  message?: string
  success: boolean
}

export interface ApiError {
  code: number
  message: string
  type: string
  details?: any
}

// 表单相关类型
export interface FormState {
  loading: boolean
  error?: string
  success?: boolean
}

// 上传相关类型
export interface UploadFile {
  uid: string
  name: string
  status: 'uploading' | 'done' | 'error' | 'removed'
  url?: string
  response?: any
  error?: any
  percent?: number
}

// 主题相关类型
export interface ThemeConfig {
  primaryColor: string
  borderRadius: number
  fontSize: number
}

// 应用配置类型
export interface AppConfig {
  apiBaseUrl: string
  uploadMaxSize: number
  supportedFileTypes: string[]
  theme: ThemeConfig
}

// 菜单相关类型
export interface MenuItem {
  key: string
  label: string
  icon?: any
  path?: string
  children?: MenuItem[]
  roles?: string[]
}

// 通知相关类型
export interface Notification {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  message: string
  duration?: number
  timestamp: number
}

// 会话相关类型
export interface ChatSession {
  id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
}

export interface ChatMessage {
  id: string
  session_id: string
  type: 'user' | 'assistant'
  content: string
  timestamp: string
  metadata?: Record<string, any>
}

// 系统健康检查类型
export interface HealthCheck {
  status: 'healthy' | 'unhealthy'
  timestamp: string
  components: {
    database: string
    vector_service: string
    redis: string
  }
  version: string
}

// 知识库相关类型
export interface KnowledgeBase {
  id: string
  name: string
  code: string
  description?: string
  category?: string
  is_active: boolean
  document_count: number
  created_by: number
  created_at: string
  updated_at: string
}

export interface KnowledgeBaseCreate {
  name: string
  code: string
  description?: string
  category?: string
  is_active?: boolean
}

export interface KnowledgeBaseUpdate {
  name?: string
  description?: string
  category?: string
  is_active?: boolean
}

export interface KnowledgeBaseSelector {
  selectedKbIds: string[]
  onSelectionChange: (kbIds: string[]) => void
  placeholder?: string
  maxCount?: number
  disabled?: boolean
}