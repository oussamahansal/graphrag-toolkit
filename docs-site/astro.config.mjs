import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import react from '@astrojs/react';

// `site` and `base` together determine the public URL.
// For the upstream awslabs project page:  site='https://awslabs.github.io', base='/graphrag-toolkit'
// For a fork's project page:               site='https://<user>.github.io',  base='/<repo-name>'
export default defineConfig({
  site: 'https://oussamahansal.github.io',
  base: '/graphrag-toolkit-fork1',
  integrations: [
    react(),
    starlight({
      title: 'GraphRAG Toolkit',
      description:
        'Documentation for the AWS GraphRAG Toolkit — lexical-graph and BYOKG-RAG.',
      logo: { src: './src/assets/logo.svg' },
      customCss: ['./src/styles/custom.css'],
      social: {
        github: 'https://github.com/awslabs/graphrag-toolkit',
      },
      head: [
        {
          tag: 'link',
          attrs: {
            rel: 'stylesheet',
            href: 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap',
          },
        },
      ],
      sidebar: [
        {
          label: 'Lexical Graph',
          items: [
            { label: 'Overview', slug: 'lexical-graph/overview' },
            { label: 'Indexing', slug: 'lexical-graph/indexing' },
            { label: 'Querying', slug: 'lexical-graph/querying' },
            { label: 'Configuration', slug: 'lexical-graph/configuration' },
          ],
        },
        {
          label: 'BYOKG-RAG',
          items: [
            { label: 'Overview', slug: 'byokg-rag/overview' },
            { label: 'Query Engine', slug: 'byokg-rag/query-engine' },
            { label: 'Configuration', slug: 'byokg-rag/configuration' },
          ],
        },
      ],
    }),
  ],
});
