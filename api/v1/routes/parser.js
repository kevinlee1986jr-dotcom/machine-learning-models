// parser.js

const fs = require('fs');
const path = require('path');
const { promisify } = require('util');
const { console } = console;

const readDirAsync = promisify(fs.readdir);
const readFileAsync = promisify(fs.readFile);
const statAsync = promisify(fs.stat);

class Parser {
  constructor(rootDir) {
    this.rootDir = rootDir;
    this.models = {};
  }

  async loadModels() {
    const files = await readDirAsync(this.rootDir);
    for (const file of files) {
      const filePath = path.join(this.rootDir, file);
      const stat = await statAsync(filePath);
      if (stat.isDirectory()) {
        this.models[file] = new Parser(filePath);
        await this.models[file].loadModels();
      } else if (file.endsWith('.json')) {
        const data = await readFileAsync(filePath, 'utf8');
        const model = JSON.parse(data);
        if (!this.models[model.name]) {
          this.models[model.name] = model;
        }
      }
    }
    return this.models;
  }
}

module.exports = Parser;