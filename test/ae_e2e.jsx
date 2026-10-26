// After Effects end-to-end check for Real Motion Blur. Run on an EMPTY project (it refuses otherwise):
//   osascript -e 'tell application "Adobe After Effects 2026" to DoScriptFile "<path>/test/ae_e2e.jsx"'
// Builds an opaque noise-textured square moving 30 px/frame over a noise background, applies the effect in each mode, renders frame 12 of every
// variant to build/ae_e2e/*.png (Metal GPU and software, 8, 16 and 32 bpc), plus Image Track off and the Vectors effect,
// and logs a one-line summary per render.
(function () {
    if (app.project.numItems > 0) return "ABORT: open an empty project first (this test builds its own comps)";
    var out = new Folder(Folder($.fileName).parent.parent.fsName + "/build/ae_e2e");
    if (!out.exists) out.create();
    var proj = app.project, log = [];
    app.beginSuppressDialogs();

    function movingSquare(comp, name) {
        var sq = comp.layers.addSolid([1, 0.6, 0.2], name, 300, 300, 1);
        // opaque, non-repeating texture that moves with the layer (Checkerboard would be half transparent and periodic)
        var tex = sq.Effects.addProperty("ADBE Fractal Noise");
        tex.property("Brightness").setValue(35);
        tex.property("Contrast").setValue(160);
        sq.property("Position").setValueAtTime(0, [300, 540]);
        sq.property("Position").setValueAtTime(2, [300 + 30 * 48, 540]);  // 30 px/frame at 24 fps
        return sq;
    }
    var scene = proj.items.addComp("scene", 1920, 1080, 1, 3, 24);
    scene.layers.addSolid([0.3, 0.3, 0.3], "bg", 1920, 1080, 1).Effects.addProperty("ADBE Fractal Noise");
    movingSquare(scene, "fg");
    var matte = proj.items.addComp("matte", 1920, 1080, 1, 3, 24);  // the square alone, on transparency
    movingSquare(matte, "fg");

    function variant(name, legacy, useMatte) {
        var c = proj.items.addComp("rmb " + name, 1920, 1080, 1, 3, 24);
        var m = c.layers.add(matte);
        m.enabled = false;
        var l = c.layers.add(scene);
        var fx = l.Effects.addProperty("RPLT RealMotionBlur");
        if (!fx) throw new Error("effect not found: is RealMotionBlur.plugin installed?");
        fx.property("Legacy").setValue(legacy ? 1 : 0);
        fx.property("Blur Amount").setValue(0.5);
        if (useMatte) {
            fx.property("FG1 Matte").setValue(m.index);
            fx.property("FG1 Matte Chan").setValue(4);  // Alpha
        }
        return c;
    }
    function render(comp, file, gpu, bpc) {
        while (proj.renderQueue.numItems) proj.renderQueue.item(1).remove();
        proj.gpuAccelType = gpu ? GpuAccelType.METAL : GpuAccelType.SOFTWARE;
        proj.bitsPerChannel = bpc;
        var rqi = proj.renderQueue.items.add(comp);
        rqi.timeSpanStart = 12 / 24;
        rqi.timeSpanDuration = 1 / 24;
        var om = rqi.outputModule(1), ext = "png";
        try {
            om.setSettings({ "Format": "PNG Sequence" });
        } catch (e) {  // fall back to any TIFF template
            var tpl = null;
            for (var i = 0; i < om.templates.length; i++) if (/tiff/i.test(om.templates[i]) && !tpl) tpl = om.templates[i];
            if (!tpl) throw new Error("no PNG/TIFF output: " + e.toString());
            om.applyTemplate(tpl);
            ext = "tif";
        }
        om.file = new File(out.fsName + "/" + file + "_[#####]." + ext);
        var t0 = new Date().getTime();
        proj.renderQueue.render();
        log.push(file + ": " + (new Date().getTime() - t0) + " ms (" + (gpu ? "Metal" : "software") + ", " + bpc + " bpc)");
    }

    try {
        var fxParams = [], probe = variant("probe", false, false), fx = probe.layer(1).Effects.property(1);
        for (var i = 1; i <= fx.numProperties; i++) fxParams.push(fx.property(i).name);
        log.push("params: " + fxParams.join(", "));
        render(scene, "scene_ref", true, 8);
        render(variant("new", false, false), "new_gpu8", true, 8);
        render(variant("legacy", true, false), "legacy_gpu8", true, 8);
        render(variant("new matte", false, true), "new_matte_gpu8", true, 8);
        render(variant("legacy matte", true, true), "legacy_matte_gpu8", true, 8);
        render(variant("new sw", false, false), "new_sw8", false, 8);
        render(variant("legacy sw32", true, false), "legacy_sw32", false, 32);
        render(variant("new gpu32", false, false), "new_gpu32", true, 32);
        render(variant("legacy matte sw16", true, true), "legacy_matte_sw16", false, 16);
        render(variant("new gpu16", false, false), "new_gpu16", true, 16);
        // Image Track off with no track points or masks: no motion, so the frame comes through unblurred.
        var still = variant("legacy no image track", true, false);
        still.layer(1).Effects.property(1).property("Image Track").setValue(0);
        render(still, "legacy_noimagetrack_gpu8", true, 8);
        // Vectors effect: a flat vector layer saying +30 px in x blurs the whole scene horizontally.
        var vc = proj.items.addComp("rmb vectors", 1920, 1080, 1, 3, 24);
        var vl = vc.layers.addSolid([0.5 + 30 / 128, 0.5, 0], "vectors", 1920, 1080, 1);
        vl.enabled = false;
        var vfx = vc.layers.add(scene).Effects.addProperty("RPLT RealMotionBlurVectors");
        if (!vfx) throw new Error("Vectors effect not found");
        vfx.property("Motion Vectors").setValue(vl.index);
        vfx.property("Blur Amt").setValue(1);
        render(vc, "vectors_gpu8", true, 8);
        render(vc, "vectors_sw8", false, 8);
    } catch (e) {
        log.push("ERROR: " + e.toString());
    }
    app.endSuppressDialogs(false);
    // AppleScript's DoScript does not relay return values, so leave the log as comp names in a saved copy of the
    // test project: build/ae_e2e/e2e.aep (read it with `strings`).
    for (var j = 0; j < log.length; j++) proj.items.addComp("LOG " + j + " " + log[j].substring(0, 230), 16, 16, 1, 1, 24);
    proj.save(new File(out.fsName + "/e2e.aep"));
    return log.join("\n");
})();
