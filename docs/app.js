// OS-aware download + gentle scroll reveal. No dependencies.
(function () {
  "use strict";

  // Enable JS-only behaviours (the scroll-reveal CSS is scoped to .js so content
  // stays visible if scripting is unavailable).
  document.documentElement.classList.add("js");

  // Single source of truth for the release links.
  var REPO = "https://github.com/Phhofm/heart-nerve_trainner";
  var assets = {
    windows: REPO + "/releases/latest/download/HeartNerveTrainer-Setup.exe",
    linux: REPO + "/releases/latest/download/HeartNerveTrainer.AppImage",
  };

  function detectOS() {
    var ua = (navigator.userAgent || "").toLowerCase();
    var platform = (navigator.userAgentData && navigator.userAgentData.platform || navigator.platform || "").toLowerCase();
    if (ua.indexOf("windows") !== -1 || platform.indexOf("win") !== -1) return "windows";
    if (ua.indexOf("linux") !== -1 || ua.indexOf("x11") !== -1 || platform.indexOf("linux") !== -1) return "linux";
    return null;
  }

  // Keep hrefs authoritative even if someone edits the HTML.
  document.querySelectorAll(".btn[data-os]").forEach(function (el) {
    var os = el.getAttribute("data-os");
    if (assets[os]) el.href = assets[os];
  });

  // Highlight (and float to the front) the button for the visitor's OS.
  var os = detectOS();
  if (os) {
    document.querySelectorAll(".btn[data-os]").forEach(function (el) {
      if (el.getAttribute("data-os") === os) el.classList.add("primary");
      else el.classList.add("dim");
    });
  }

  // Reveal-on-scroll, respecting reduced motion.
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var targets = document.querySelectorAll(".reveal");
  if (reduce || !("IntersectionObserver" in window)) {
    targets.forEach(function (el) { el.classList.add("in"); });
  } else {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
      });
    }, { threshold: 0.14 });
    targets.forEach(function (el) { io.observe(el); });
  }
})();
