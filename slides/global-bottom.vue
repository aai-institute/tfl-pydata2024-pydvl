<script setup>
import { onMounted } from "vue";

onMounted(() => {
  const script = document.createElement("script");
  script.async = true;
  script.defer = true;
  script.src = "https://scripts.simpleanalyticscdn.com/latest.js";
  document.head.appendChild(script);
});
</script>

<template>
  <div v-if="$slidev.nav.currentLayout !== 'cover'">
    <div
      class="absolute text-xs bottom-0 right-0 w-full flex justify-center items-center"
    >
      <span><SlideCurrentNo /> / <SlidesTotal /></span>
    </div>
    <a href="https://transferlab.ai" target="_blank">
      <img
        class="absolute bottom-1 right-1 w-40 z-9999"
        src="/transferlab-logo.svg"
        alt="TransferLab logo"
      />
    </a>
  </div>
</template>
