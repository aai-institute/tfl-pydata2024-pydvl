<template>
  <div v-if="$slidev.nav.currentLayout !== 'cover'">
    <div class="absolute text-xs bottom-0 right-0 w-full flex justify-center items-center">
        <span><SlideCurrentNo /> / <SlidesTotal /></span>
    </div>
    <img class="absolute bottom-1 right-1 w-40 z-9999"
        src="/transferlab-logo.svg"
        alt="TransferLab logo" />
  </div>
</template>