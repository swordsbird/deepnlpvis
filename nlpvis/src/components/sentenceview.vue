<template>
  <v-simple-table dense style="overflow-y: scroll; max-width: 100%;">
    <template v-slot:default>
      <thead>
        <tr>
          <th class="text-left pa-1" style="white-space:nowrap;">
            id
            <v-icon size="12" style="color: #444">mdi-menu-swap</v-icon>
          </th>
          <th class="text-left pa-1">
            passage
            <v-icon size="12" style="color: #444">mdi-menu-swap</v-icon>
          </th>
          <th class="text-left pa-1" style="white-space:nowrap;">
            label
            <v-icon size="12" style="color: #444">mdi-menu-swap</v-icon>
          </th>
          <th class="text-left pa-1" style="white-space:nowrap;">
            score
            <v-icon size="12" style="color: #444">mdi-menu-swap</v-icon>
          </th>
          <th class="text-left pa-1" style="white-space:nowrap;">
            correctness
            <v-icon size="12" style="color: #444">mdi-menu-swap</v-icon>
          </th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="(item) in samples"
          :key="item.index"
          @mouseenter="highlightGrid([item.index])"
          @mouseleave="highlightGrid(null)"
          @click="addDAG(item.index)"
        >
          <td class="pa-1">{{ item.index }}</td>
          <td class="pa-1">
            <span
              v-for="(word, index) in item.text"
              :key="index"
              :style="{ background: word[1] ? 'orange' : 'none' }"
            >{{ word[0] }}</span>
          </td>
          <td class="pa-1" style="white-space:nowrap;">
            {{ item.label }}
            <span
              style="display: inline-block;width:12px; height:12px; border:.5px solid lightgray; opacity: .7;"
              :style="{ 'background-color': item.label == 0 ? color[0] : color[1] }"
            ></span>
          </td>
          <td class="pa-1">{{ Number(item.score).toFixed(2) }}</td>
          <td class="pa-1">
            <v-icon v-if="(item.score > 0.5) == item.label">mdi-check</v-icon>
            <v-icon v-else>mdi-cross</v-icon>
          </td>
        </tr>
      </tbody>
    </template>
  </v-simple-table>
</template>
<script>
import bus from '../plugins/bus'

export default {
  data(){
    return {}
  },
  props: [ 'samples', 'color' ],
  methods: {
    highlightGrid(idxs) {
      bus.$emit('highlight_grid', idxs)
    },
    addDAG(idx) {
      bus.$on('add_sentence_DAG', idx)
    }
  }
}
</script>