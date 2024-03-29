// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const path = require('node:path');
const CopyPlugin = require('copy-webpack-plugin');

module.exports = {
  target : ['web'],
  entry : path.resolve(__dirname, 'src/cjs-js/main.js'),
  output : {
    clean : true,
    filename : 'ort-test-e2e.bundle.js',
    path : path.resolve(__dirname, 'dist/webpack_umd_js'),
    library : {type : 'umd'},
  },
  plugins :
  [
    new CopyPlugin({patterns : [{from : 'node_modules/onnxruntime-web/dist/ort-*.{js,wasm}', to : '[name][ext]'}]}),
  ]
};
