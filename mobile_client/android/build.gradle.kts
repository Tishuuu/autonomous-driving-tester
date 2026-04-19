allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
    
}


tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}

subprojects {
    val subproject = this
    subproject.plugins.whenPluginAdded {
        if (this.javaClass.name.contains("com.android.build.gradle.LibraryPlugin")) {
            val android = subproject.extensions.getByName("android")
            try {
                val setNamespace = android.javaClass.getMethod("setNamespace", String::class.java)
                // יצירת Namespace ייחודי לכל חבילה (למשל com.fixed.sensors_plus)
                val uniqueNamespace = "com.fixed.${subproject.name.replace("-", "_")}"
                setNamespace.invoke(android, uniqueNamespace)
                println("🛡️ Unique Fix: Injected [$uniqueNamespace] into ${subproject.name}")
            } catch (e: Exception) { }
        }
    }
}