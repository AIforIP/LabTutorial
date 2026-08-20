import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'zh-CN',
  title: 'Lab Tutorial',
  description: '面向中文学生的计算机科学与人工智能实践教程',
  base: '/',
  cleanUrls: true,
  lastUpdated: true,
  markdown: {
    math: true
  },
  head: [
    ['meta', { name: 'theme-color', content: '#fafaf8' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }]
  ],
  themeConfig: {
    logo: false,
    siteTitle: 'Lab Tutorial',
    nav: [
      { text: '首页', link: '/' },
      { text: 'GitHub', link: 'https://github.com/AIforIP' }
    ],
    sidebar: {
      '/before-llm/': [
        {
          text: 'Before LLM',
          items: [
            { text: '学习导览', link: '/before-llm/' },
            { text: '1. 科研基本功', link: '/before-llm/research' },
            { text: '2. Python 与开发环境', link: '/before-llm/programming' }
          ]
        },
        {
          text: '模型基础',
          items: [
            { text: '3. 机器学习', link: '/before-llm/machine-learning' },
            { text: '4. 深度学习', link: '/before-llm/deep-learning' },
            { text: '5. NLP 基础', link: '/before-llm/nlp' }
          ]
        }
      ],
      '/llm/': [
        {
          text: 'LLM from Scratch',
          items: [
            { text: '课程导览', link: '/llm/' },
            { text: '0. 准备工作', link: '/llm/preparation' }
          ]
        },
        {
          text: '第一部分 · 基础组件',
          items: [
            { text: '1. BPE Tokenizer', link: '/llm/tokenizer' },
            { text: '2. Transformer', link: '/llm/transformer' }
          ]
        },
        {
          text: '第二部分 · 训练系统',
          items: [
            { text: '3. 训练一个语言模型', link: '/llm/training' }
          ]
        }
      ]
    },
    outline: {
      level: [2, 3],
      label: '本页目录'
    },
    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    },
    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'medium'
      }
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/AIforIP' }
    ],
    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索', buttonAriaLabel: '搜索文档' },
          modal: {
            noResultsText: '没有找到相关内容',
            resetButtonTitle: '清除查询',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' }
          }
        }
      }
    }
  }
})
