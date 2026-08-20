---
layout: doc
pageClass: home-page
sidebar: false
aside: false
title: 首页
lastUpdated: false
---

<script setup>
import { computed, ref } from 'vue'

const tutorials = [
  {
    title: 'Before LLM：大语言模型前的基础知识',
    description: '从科研工具、Python 编程到机器学习、深度学习与 NLP，建立进入大模型研究前的基础能力。',
    href: './before-llm/',
    tags: ['入门', '科研训练', '机器学习', '深度学习', 'NLP'],
    status: '持续更新',
    date: '2026.08'
  },
  {
    title: 'LLM from Scratch：从零构建大语言模型',
    description: '从 Tokenizer、Transformer 到训练系统，理解语言模型如何从原始文本一步步建立起来。',
    href: './llm/',
    tags: ['LLM', 'CS336', '基础与实践'],
    status: '持续更新',
    date: '2026.08'
  }
]

const activeTag = ref('')

const tagCounts = computed(() => {
  const counts = new Map()
  for (const tutorial of tutorials) {
    for (const tag of tutorial.tags) {
      counts.set(tag, (counts.get(tag) || 0) + 1)
    }
  }
  return [...counts.entries()]
})

const filteredTutorials = computed(() => {
  if (!activeTag.value) return tutorials
  return tutorials.filter((tutorial) => tutorial.tags.includes(activeTag.value))
})

function toggleTag(tag) {
  activeTag.value = activeTag.value === tag ? '' : tag
}

function openSearch() {
  document.querySelector('.VPNavBarSearch button')?.click()
}
</script>

<div class="home-layout">
<aside class="tag-sidebar">
<strong>标签</strong>
<button
  v-for="([tag, count]) in tagCounts"
  :key="tag"
  type="button"
  :class="{ active: activeTag === tag }"
  @click="toggleTag(tag)"
><span>{{ tag }}</span><span>{{ count }}</span></button>
</aside>
<main class="home-main">
<div class="home-header">
<h1>学习材料</h1>
<p>IP Intelligence Lab 学习材料。</p>
</div>
<button class="home-search" type="button" @click="openSearch">
<span class="home-search-icon" aria-hidden="true"></span>
<span>搜索教程与章节</span>
<kbd>Ctrl K</kbd>
</button>
<div class="list-heading">
<h2>{{ activeTag ? `#${activeTag}` : '全部教程' }}</h2>
<span>{{ filteredTutorials.length }} 篇</span>
</div>
<div class="material-list">
<a
  v-for="tutorial in filteredTutorials"
  :key="tutorial.href"
  class="material-entry"
  :href="tutorial.href"
>
<span class="material-main">
<span class="material-title">{{ tutorial.title }}</span>
<span class="material-desc">{{ tutorial.description }}</span>
<span class="material-meta"><span v-for="tag in tutorial.tags" :key="tag">#{{ tag }}</span></span>
</span>
<span class="material-side"><span>{{ tutorial.status }}</span><span>{{ tutorial.date }}</span></span>
</a>
</div>
</main>
</div>
