import React, { useState, useEffect } from 'react'
import { Select, Tag, Input, Space, Spin, Empty, message } from 'antd'
import { SearchOutlined, DatabaseOutlined } from '@ant-design/icons'
import { KnowledgeBase, KnowledgeBaseSelector as KnowledgeBaseSelectorProps } from '../types'
import { getActiveKnowledgeBases } from '../services/kbApi'

const { Option } = Select

interface KnowledgeBaseSelectorComponentProps extends KnowledgeBaseSelectorProps {
  className?: string
  style?: React.CSSProperties
  size?: 'small' | 'middle' | 'large'
  allowClear?: boolean
  showSearch?: boolean
}

const KnowledgeBaseSelector: React.FC<KnowledgeBaseSelectorComponentProps> = ({
  selectedKbIds,
  onSelectionChange,
  placeholder = '选择知识库',
  maxCount,
  disabled = false,
  className,
  style,
  size = 'middle',
  allowClear = true,
  showSearch = true
}) => {
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([])
  const [loading, setLoading] = useState(false)
  const [searchValue, setSearchValue] = useState('')

  // 加载知识库列表
  const loadKnowledgeBases = async () => {
    setLoading(true)
    try {
      const kbs = await getActiveKnowledgeBases()
      setKnowledgeBases(kbs)
    } catch (error) {
      console.error('Failed to load knowledge bases:', error)
      message.error('加载知识库列表失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadKnowledgeBases()
  }, [])

  // 过滤知识库
  const filteredKnowledgeBases = (knowledgeBases || []).filter(kb => 
    kb.name.toLowerCase().includes(searchValue.toLowerCase()) ||
    (kb.description && kb.description.toLowerCase().includes(searchValue.toLowerCase())) ||
    (kb.category && kb.category.toLowerCase().includes(searchValue.toLowerCase()))
  )

  // 处理选择变化
  const handleChange = (values: string[]) => {
    onSelectionChange(values)
  }

  // 自定义标签渲染
  const tagRender = (props: any) => {
    const { label, value, closable, onClose } = props
    const kb = knowledgeBases.find(k => k.id === value)
    
    return (
      <Tag
        color="blue"
        closable={closable}
        onClose={onClose}
        style={{ marginRight: 3 }}
        icon={<DatabaseOutlined />}
      >
        {kb?.name || label}
      </Tag>
    )
  }

  // 自定义下拉选项渲染
  const renderOption = (kb: KnowledgeBase) => (
    <Option key={kb.id} value={kb.id}>
      <Space>
        <DatabaseOutlined style={{ color: '#1890ff' }} />
        <div>
          <div style={{ fontWeight: 500 }}>{kb.name}</div>
          {kb.description && (
            <div style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>
              {kb.description}
            </div>
          )}
          <div style={{ fontSize: '11px', color: '#999', marginTop: '2px' }}>
            {kb.category && `分类: ${kb.category} • `}
            文档数: {kb.document_count}
          </div>
        </div>
      </Space>
    </Option>
  )

  return (
    <Select
      mode="multiple"
      value={selectedKbIds}
      onChange={handleChange}
      placeholder={placeholder}
      disabled={disabled}
      loading={loading}
      allowClear={allowClear}
      showSearch={showSearch}
      maxCount={maxCount}
      className={className}
      style={style}
      size={size}
      tagRender={tagRender}
      searchValue={searchValue}
      onSearch={setSearchValue}
      filterOption={false}
      dropdownStyle={{ maxWidth: 300 }}
      notFoundContent={
        loading ? (
          <div style={{ textAlign: 'center', padding: '20px' }}>
            <Spin size="small" />
            <div style={{ marginTop: '8px' }}>加载中...</div>
          </div>
        ) : filteredKnowledgeBases.length === 0 ? (
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description="暂无知识库"
            style={{ padding: '20px' }}
          />
        ) : null
      }
      popupRender={(menu) => (
        <div>
          {showSearch && (
            <div style={{ padding: '8px' }}>
              <Input
                prefix={<SearchOutlined />}
                placeholder="搜索知识库"
                value={searchValue}
                onChange={(e) => setSearchValue(e.target.value)}
                style={{ width: '100%' }}
              />
            </div>
          )}
          {menu}
        </div>
      )}
    >
      {filteredKnowledgeBases.map(renderOption)}
    </Select>
  )
}

export default KnowledgeBaseSelector