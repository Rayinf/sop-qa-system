import { useState, useEffect } from 'react'
import {
  Row,
  Col,
  Card,
  Statistic,
  Typography,
  Space,
  Button,
  List,
  Avatar,
  Tag,
  Progress,
  Divider,
  message,
  Spin
} from 'antd'
import {
  QuestionCircleOutlined,
  FileTextOutlined,
  UserOutlined,
  ClockCircleOutlined,
  TrophyOutlined,
  RightOutlined
} from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import { QAStatistics, DocumentStatistics, PersonalQAStatistics } from '../types/index'
import { StatisticsApi } from '../services/statisticsApi'

const { Title, Text } = Typography

// 计算用户满意度（基于反馈分布）
const calculateSatisfactionRate = (feedbackDistribution: any) => {
  if (!feedbackDistribution) return 0
  const total = Object.values(feedbackDistribution).reduce((sum: number, count: any) => sum + (count || 0), 0)
  if (total === 0) return 0
  const positive = (feedbackDistribution.helpful || 0) + (feedbackDistribution.very_helpful || 0)
  return (positive / total) * 100
}

interface DashboardProps {
  user?: any
}

const Dashboard = ({ user }: DashboardProps) => {
  const navigate = useNavigate()
  const [qaStats, setQaStats] = useState<QAStatistics | null>(null)
  const [docStats, setDocStats] = useState<DocumentStatistics | null>(null)
  const [personalStats, setPersonalStats] = useState<PersonalQAStatistics | null>(null)
  const [recentQuestions, setRecentQuestions] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  // 加载统计数据
   const loadStats = async () => {
     setLoading(true)
     try {
       const { qaStats: qaStatsData, documentStats: docStatsData, personalStats: personalStatsData, recentQA } = await StatisticsApi.getDashboardStatistics()
       
       setQaStats(qaStatsData)
       setDocStats(docStatsData)
       setPersonalStats(personalStatsData)
       
       // 转换QA历史数据格式
       const formattedRecentQuestions = recentQA.map(qa => ({
         id: qa.id,
         question: qa.question,
         answer: qa.answer,
         created_at: qa.created_at,
         confidence: 0.85 + Math.random() * 0.15 // 模拟置信度，实际应该从API获取
       }))
       setRecentQuestions(formattedRecentQuestions)
     } catch (error) {
       console.error('加载统计数据失败:', error)
       message.error('加载统计数据失败，请稍后重试')
     } finally {
       setLoading(false)
     }
   }

  useEffect(() => {
    loadStats()
  }, [])

  // 最近问答记录现在从API获取，存储在state中

  // 格式化文件大小
  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  // 格式化时间
  const formatTime = (dateString: string) => {
    const date = new Date(dateString)
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const hours = Math.floor(diff / (1000 * 60 * 60))
    
    if (hours < 1) {
      const minutes = Math.floor(diff / (1000 * 60))
      return `${minutes}分钟前`
    } else if (hours < 24) {
      return `${hours}小时前`
    } else {
      const days = Math.floor(hours / 24)
      return `${days}天前`
    }
  }

  return (
    <div>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>仪表板</Title>
        <Text type="secondary">
          欢迎回来，{user?.full_name || '用户'}！这里是您的系统概览。
        </Text>
      </div>

      {/* 统计卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="总问题数"
              value={qaStats?.total_questions || 0}
              prefix={<QuestionCircleOutlined />}
              loading={loading}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="文档总数"
              value={docStats?.total_documents || 0}
              prefix={<FileTextOutlined />}
              loading={loading}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="今日问答"
              value={qaStats?.today_questions || 0}
              prefix={<UserOutlined />}
              loading={loading}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="平均响应时间"
              value={qaStats?.average_processing_time || 0}
              suffix="秒"
              precision={1}
              prefix={<ClockCircleOutlined />}
              loading={loading}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        {/* 系统状态 */}
        <Col xs={24} lg={12}>
          <Card title="系统状态" loading={loading}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                  <Text>文档处理进度</Text>
                  <Text>{docStats ? Math.round(((docStats.status_distribution.vectorized || 0) / docStats.total_documents) * 100) : 0}%</Text>
                </div>
                <Progress 
                  percent={docStats ? Math.round(((docStats.status_distribution.vectorized || 0) / docStats.total_documents) * 100) : 0}
                  status="active"
                />
              </div>
              
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                  <Text>用户满意度</Text>
                  <Text>{qaStats ? Math.round(calculateSatisfactionRate(qaStats.feedback_distribution)) : 0}%</Text>
                </div>
                <Progress 
                  percent={qaStats ? Math.round(calculateSatisfactionRate(qaStats.feedback_distribution)) : 0}
                  strokeColor="#52c41a"
                />
              </div>
              
              <Divider style={{ margin: '12px 0' }} />
              
              <Row gutter={16}>
                <Col span={12}>
                  <Statistic
                    title="处理中文档"
                    value={(docStats?.status_distribution.processing || 0) + (docStats?.status_distribution.uploaded || 0)}
                    valueStyle={{ fontSize: '16px' }}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="已完成文档"
                    value={docStats?.status_distribution.vectorized || 0}
                    valueStyle={{ fontSize: '16px' }}
                  />
                </Col>
              </Row>
            </Space>
          </Card>
        </Col>

        {/* 最近问答 */}
        <Col xs={24} lg={12}>
          <Card 
            title="最近问答" 
            loading={loading}
            extra={
              <Button 
                type="link" 
                icon={<RightOutlined />}
                onClick={() => navigate('/qa')}
              >
                查看全部
              </Button>
            }
          >
            <List
              dataSource={recentQuestions}
              renderItem={(item) => (
                <List.Item>
                  <List.Item.Meta
                    avatar={
                      <Avatar 
                        icon={<QuestionCircleOutlined />} 
                        style={{ backgroundColor: '#1890ff' }}
                      />
                    }
                    title={
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Text strong style={{ fontSize: '14px' }}>
                          {item.question.length > 30 ? item.question.substring(0, 30) + '...' : item.question}
                        </Text>
                        <Tag color={item.confidence > 0.9 ? 'green' : item.confidence > 0.8 ? 'blue' : 'orange'}>
                          <TrophyOutlined /> {Math.round(item.confidence * 100)}%
                        </Tag>
                      </div>
                    }
                    description={
                      <div>
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          {item.answer.length > 50 ? item.answer.substring(0, 50) + '...' : item.answer}
                        </Text>
                        <br />
                        <Text type="secondary" style={{ fontSize: '11px' }}>
                          {formatTime(item.created_at)}
                        </Text>
                      </div>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </Col>
      </Row>

      {/* 快速操作 */}
      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col span={24}>
          <Card title="快速操作">
            <Space wrap>
              <Button 
                type="primary" 
                icon={<QuestionCircleOutlined />}
                onClick={() => navigate('/qa')}
              >
                开始问答
              </Button>
              <Button 
                icon={<FileTextOutlined />}
                onClick={() => navigate('/documents')}
              >
                管理文档
              </Button>
              <Button 
                icon={<UserOutlined />}
                onClick={() => navigate('/profile')}
              >
                个人设置
              </Button>
              {user?.role === 'admin' && (
                <Button 
                  icon={<TrophyOutlined />}
                  onClick={() => navigate('/admin')}
                >
                  系统管理
                </Button>
              )}
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default Dashboard