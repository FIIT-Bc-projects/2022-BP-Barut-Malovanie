<script>

import axios from 'axios'
export default {
  data() {
    return {
      input: "A white flower",
      images: []
    }
  },
  methods: {
    async submit() {
      const res =  await axios.get('http://127.0.0.1:8001', { params: { "input" : this.input } })
      this.images = res.data.images;
      console.log(this.images);
    }
  },
}
</script>

<template>
  <main>
    <div class="input-form">
      <input v-model="input" placeholder="Describe your flower" type="text" />
      <button @click="submit">Generate</button>
    </div>
    <div class="image-container">
      <div v-for="image in images">
        <img  v-bind:src="'data:image/png;base64, ' +image" alt=":("/>
      </div>
    </div>
  </main>
</template>

<style scoped>
  .input-form {
    display: flex;
    justify-content: space-around;
  }
  input {
    width: 80%;
    color: #999;
    border:none;
    background: transparent;
    border-bottom: 2px solid #282828;
    font-size: xx-large;
   }
  input:focus {
    border: none;
    outline: none;
    border-bottom: 2px solid #aaaaaa;
  }
  .image-container {
    width: auto;
    display: flex;
    justify-content: space-between;
    margin-top: 50px;
  }
  img {
    height: 250px;
  }
  button {
    color: #aaa;
    background: #282828;
    border-radius: 5px;
    border: none;
    font-size: x-large;
    padding: 7px;
    margin-bottom: -5px;
  }
</style>
