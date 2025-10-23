const fs = require('fs');
const assert = require('assert');

const source = fs.readFileSync('app/static/main.js', 'utf8');

const setMatch = source.match(/const MATH_LINEBREAK_ENVS = new Set\([\s\S]*?\);/);
const ensureMatch = source.match(/function ensureMathLineBreaks\(math\) {([\s\S]*?)^}/m);
const sanitizeMatch = source.match(/function sanitizeMathContent\(source\) {([\s\S]*?)^}/m);

if (!setMatch || !ensureMatch || !sanitizeMatch) {
  throw new Error('failed to extract math helpers');
}

const buildModule = new Function(
  `${setMatch[0]}\n` +
  `function ensureMathLineBreaks(math) {${ensureMatch[1]}}\n` +
  `function sanitizeMathContent(source) {${sanitizeMatch[1]}}\n` +
  'return { sanitizeMathContent, ensureMathLineBreaks, MATH_LINEBREAK_ENVS };'
);

const { sanitizeMathContent } = buildModule();

const alignedSnippet = String.raw`\[ \begin{aligned} M(\theta) &= \mathcal{A}^T \mathcal{L}^T(\theta) \mathcal{G} \mathcal{L}(\theta) \mathcal{A}, \\ c(\theta, \dot{\theta}) &= -\mathcal{A}^T \mathcal{L}^T(\theta) \left( \mathcal{G} \mathcal{L}(\theta) \left[ \text{ad}_{\mathcal{A} \dot{\theta}} \right] \mathcal{W}(\theta) + \left[ \text{ad}_{\mathcal{V}} \right]^T \mathcal{G} \right) \mathcal{L}(\theta) \mathcal{A} \dot{\theta} \\ g(\theta) &= \mathcal{A}^T \mathcal{L}^T(\theta) \mathcal{g} \mathcal{L}(\theta) \dot{\mathcal{V}}_{\text{base}}. \end{aligned} \]`;

const sanitized = sanitizeMathContent(alignedSnippet);
const lineBreaks = (sanitized.match(/\\\\/g) || []).length;
assert.strictEqual(lineBreaks, 2, 'aligned block should contain three explicit \\ line breaks');

console.log('math linebreak test passed');
