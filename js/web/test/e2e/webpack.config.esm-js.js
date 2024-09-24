// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const path = require('node:path');

module.exports = {
  experiments: { outputModule: true },
  target: ['web'],
  entry: path.resolve(__dirname, 'src/esm-js/main.js'),
  output: {
    clean: true,
    filename: 'ort-test-e2e.bundle.mjs',
    path: path.resolve(__dirname, 'dist/webpack_esm_js'),
    library: { type: 'module' },
    chunkFormat: 'module',
  },
};
