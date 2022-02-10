const webpack = require('webpack');

module.exports = {
  configureWebpack: {
    plugins: [
      new webpack.ProvidePlugin({
        'introJs': ['intro.js']
      })
    ]
  },
  "transpileDependencies": [
    "vuetify"
  ]
}