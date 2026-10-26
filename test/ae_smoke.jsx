// AE smoke test: textured square moving 30 px/frame in a precomp, Real Motion Blur on the precomp layer.
// Renders frame 12 of inner (reference) and outer (blurred), GPU then CPU. Returns a log string.
(function () {
    var out = Folder($.fileName).parent.parent.fsName + "/build/ae_test";
    var log = [];
    app.beginSuppressDialogs();
    var proj = app.newProject();
    var inner = proj.items.addComp("inner", 960, 540, 1, 2, 24);
    inner.layers.addSolid([0.2, 0.2, 0.2], "bg", 960, 540, 1).Effects.addProperty("ADBE Fractal Noise");
    var sq = inner.layers.addSolid([1, 0.5, 0.1], "sq", 200, 200, 1);
    sq.Effects.addProperty("ADBE Checkerboard");
    sq.property("Position").setValueAtTime(0, [100, 270]);
    sq.property("Position").setValueAtTime(1, [820, 270]);
    var outer = proj.items.addComp("outer", 960, 540, 1, 2, 24);
    var fx = outer.layers.add(inner).Effects.addProperty("RPLT RealMotionBlur");
    if (!fx) return "effect failed to apply";
    for (var i = 1; i <= fx.numProperties; i++) log.push("param: " + fx.property(i).name);
    fx.property("Blur Amount").setValue(1);

    function render(comp, name) {
        var rqi = proj.renderQueue.items.add(comp);
        rqi.timeSpanStart = 12 / 24;
        rqi.timeSpanDuration = 1 / 24;
        var om = rqi.outputModule(1);
        om.applyTemplate("PNG Sequence");
        om.file = new File(out + "/" + name + "_[#####].png");
    }
    render(inner, "ref");
    render(outer, "gpu");
    proj.gpuAccelType = GpuAccelType.METAL;
    var t0 = new Date().getTime();
    proj.renderQueue.render();
    log.push("metal render ms (2 items): " + (new Date().getTime() - t0));
    while (proj.renderQueue.numItems) proj.renderQueue.item(1).remove();
    render(outer, "cpu");
    proj.gpuAccelType = GpuAccelType.SOFTWARE;
    t0 = new Date().getTime();
    proj.renderQueue.render();
    log.push("software render ms: " + (new Date().getTime() - t0));
    proj.save(new File(out + "/smoke.aep"));
    app.endSuppressDialogs(false);
    return log.join("\n");
})();
